"""
Real-time audio capture using WASAPI loopback.
Maintains a rolling 30-second buffer for the audio encoder.
"""

import time
import threading
import queue
from typing import Optional, Callable
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[AudioCapture] WARNING: sounddevice not installed. pip install sounddevice")


class AudioCapture:
    """
    Real-time audio capture using WASAPI loopback.

    Captures system audio (game sounds) and maintains a rolling buffer.
    Designed for integration with AudioEncoder (30 sec @ 16kHz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration: float = 30.0,
        device_sample_rate: int = 48000,
        blocksize: int = 1024,
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Target sample rate (16kHz for model)
            buffer_duration: Buffer length in seconds (30 sec)
            device_sample_rate: Expected device sample rate
            blocksize: Samples per audio callback
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.device_sample_rate = device_sample_rate
        self.blocksize = blocksize

        # Buffer size in samples
        self.buffer_size = int(buffer_duration * sample_rate)

        # Rolling buffer (circular)
        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_lock = threading.Lock()

        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._device_id: Optional[int] = None
        self._actual_sample_rate: int = device_sample_rate

        # Queue for incoming audio chunks
        self._audio_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None

        # Callback for new audio
        self._callback: Optional[Callable[[np.ndarray], None]] = None

        # Find loopback device
        if SOUNDDEVICE_AVAILABLE:
            self._device_id = self._find_loopback_device()
            self._actual_sample_rate = self._get_device_sample_rate()

    def _find_loopback_device(self) -> Optional[int]:
        """Find WASAPI loopback device for system audio capture."""
        devices = sd.query_devices()

        # Look for loopback device (Windows WASAPI)
        loopback_keywords = ['loopback', 'stereo mix', 'what u hear', 'wave out']

        for i, device in enumerate(devices):
            name_lower = device['name'].lower()
            if any(keyword in name_lower for keyword in loopback_keywords):
                if device['max_input_channels'] > 0:
                    print(f"[AudioCapture] Found loopback device: {device['name']}")
                    return i

        # Try default input as fallback
        try:
            default_input = sd.query_devices(kind='input')
            print(f"[AudioCapture] Using default input: {default_input['name']}")
            return None  # None means default device
        except Exception as e:
            print(f"[AudioCapture] Warning: No suitable device found: {e}")
            return None

    def _get_device_sample_rate(self) -> int:
        """Get actual sample rate of the device."""
        try:
            if self._device_id is not None:
                device_info = sd.query_devices(self._device_id)
            else:
                device_info = sd.query_devices(kind='input')
            return int(device_info['default_samplerate'])
        except Exception:
            return self.device_sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream - pushes to queue."""
        if status:
            pass  # Ignore status messages in production

        if self._running:
            self._audio_queue.put(indata.copy())

    def _writer_loop(self):
        """Background thread for processing audio chunks."""
        while self._running or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.1)
                self._process_chunk(data)
            except queue.Empty:
                continue

    def _process_chunk(self, data: np.ndarray):
        """
        Process an audio chunk: resample, convert to mono, add to buffer.

        Args:
            data: Raw audio data from device (samples, channels)
        """
        # Convert stereo to mono if needed
        if data.ndim > 1 and data.shape[1] > 1:
            audio = data.mean(axis=1)
        else:
            audio = data.flatten()

        # Resample if needed
        if self._actual_sample_rate != self.sample_rate:
            audio = self._resample(audio, self._actual_sample_rate, self.sample_rate)

        # Add to rolling buffer
        with self._buffer_lock:
            chunk_len = len(audio)

            if self._buffer_pos + chunk_len <= self.buffer_size:
                # Fits in remaining space
                self._buffer[self._buffer_pos:self._buffer_pos + chunk_len] = audio
                self._buffer_pos += chunk_len
            else:
                # Need to wrap around - shift buffer left
                shift = chunk_len
                self._buffer[:-shift] = self._buffer[shift:].copy()
                self._buffer[-chunk_len:] = audio
                self._buffer_pos = self.buffer_size

        # Notify callback if set
        if self._callback:
            self._callback(audio)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)

        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new audio chunks: callback(audio_chunk)."""
        self._callback = callback

    def start(self):
        """Start audio capture."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioCapture] sounddevice not available, running without audio")
            return

        if self._running:
            print("[AudioCapture] Already running")
            return

        self._running = True

        # Start writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._device_id,
                channels=2,  # Capture stereo, convert to mono later
                samplerate=self._actual_sample_rate,
                dtype='float32',
                blocksize=self.blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            print(f"[AudioCapture] Started (device rate: {self._actual_sample_rate} Hz)")
        except Exception as e:
            print(f"[AudioCapture] Failed to start: {e}")
            self._running = False

    def stop(self):
        """Stop audio capture."""
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None

        print("[AudioCapture] Stopped")

    def get_buffer(self) -> np.ndarray:
        """
        Get current audio buffer (30 seconds).

        Returns:
            Audio buffer (buffer_size,) normalized float32
        """
        with self._buffer_lock:
            # Return a copy to avoid threading issues
            buffer = self._buffer.copy()

        # Normalize
        max_val = np.abs(buffer).max()
        if max_val > 0:
            buffer = buffer / max_val * 0.95

        return buffer

    def get_buffer_filled(self) -> float:
        """Get fraction of buffer filled (0-1)."""
        with self._buffer_lock:
            return self._buffer_pos / self.buffer_size

    @property
    def is_running(self) -> bool:
        """Whether capture is active."""
        return self._running

    @property
    def is_available(self) -> bool:
        """Whether audio capture is available."""
        return SOUNDDEVICE_AVAILABLE


class DummyAudioCapture:
    """
    Dummy audio capture for when sounddevice is not available.
    Returns zeros for all audio queries.
    """

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 30.0, **kwargs):
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_duration * sample_rate)
        self._running = False

    def start(self):
        self._running = True
        print("[DummyAudioCapture] Running without audio (sounddevice not available)")

    def stop(self):
        self._running = False

    def get_buffer(self) -> np.ndarray:
        return np.zeros(self.buffer_size, dtype=np.float32)

    def get_buffer_filled(self) -> float:
        return 0.0

    def set_callback(self, callback):
        pass

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_available(self) -> bool:
        return False


def get_audio_capture(**kwargs) -> AudioCapture:
    """
    Factory function to get appropriate audio capture.

    Returns AudioCapture if sounddevice available, DummyAudioCapture otherwise.
    """
    if SOUNDDEVICE_AVAILABLE:
        return AudioCapture(**kwargs)
    else:
        return DummyAudioCapture(**kwargs)


def list_audio_devices():
    """List all available audio devices."""
    if not SOUNDDEVICE_AVAILABLE:
        print("sounddevice not installed")
        return

    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()

    for i, device in enumerate(devices):
        input_ch = device['max_input_channels']
        output_ch = device['max_output_channels']
        sr = int(device['default_samplerate'])

        device_type = []
        if input_ch > 0:
            device_type.append(f"IN:{input_ch}ch")
        if output_ch > 0:
            device_type.append(f"OUT:{output_ch}ch")

        print(f"[{i}] {device['name']}")
        print(f"    {', '.join(device_type)}, {sr}Hz")

        # Mark loopback devices
        name_lower = device['name'].lower()
        if any(k in name_lower for k in ['loopback', 'stereo mix', 'what u hear']):
            print(f"    *** LOOPBACK DEVICE ***")
        print()


def test_audio_capture():
    """Test audio capture functionality."""
    print("Testing AudioCapture...")

    if not SOUNDDEVICE_AVAILABLE:
        print("sounddevice not available, skipping test")
        return

    capture = AudioCapture()

    print("\nRecording for 5 seconds...")
    capture.start()
    time.sleep(5.0)
    capture.stop()

    buffer = capture.get_buffer()
    print(f"\nBuffer shape: {buffer.shape}")
    print(f"Buffer filled: {capture.get_buffer_filled() * 100:.1f}%")
    print(f"Buffer range: [{buffer.min():.3f}, {buffer.max():.3f}]")

    print("\nTest complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_audio_devices()
    else:
        test_audio_capture()
