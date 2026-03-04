"""
Real-time stereo audio capture using WASAPI loopback.
Maintains a rolling 16-second stereo buffer for StereoAudioEncoder.

Changes from old version:
  - Keeps stereo (2 channels) instead of converting to mono
  - 16 sec buffer (was 30 sec) to match StereoAudioEncoder
  - get_buffer() returns (2, buffer_size) stereo
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
    Real-time stereo audio capture using WASAPI loopback.

    Captures system audio (game sounds) and maintains a rolling stereo buffer.
    Designed for StereoAudioEncoder (16 sec stereo @ 16kHz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration: float = 16.0,
        channels: int = 2,
        device_sample_rate: int = 48000,
        blocksize: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.channels = channels
        self.device_sample_rate = device_sample_rate
        self.blocksize = blocksize

        # Buffer size in samples per channel
        self.buffer_size = int(buffer_duration * sample_rate)

        # Stereo rolling buffer (2, buffer_size)
        self._buffer = np.zeros((channels, self.buffer_size), dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_lock = threading.Lock()
        self._total_samples = 0

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

        loopback_keywords = ['loopback', 'stereo mix', 'what u hear', 'wave out']

        for i, device in enumerate(devices):
            name_lower = device['name'].lower()
            if any(keyword in name_lower for keyword in loopback_keywords):
                if device['max_input_channels'] > 0:
                    print(f"[AudioCapture] Found loopback device: {device['name']}")
                    return i

        try:
            default_input = sd.query_devices(kind='input')
            print(f"[AudioCapture] Using default input: {default_input['name']}")
            return None
        except Exception as e:
            print(f"[AudioCapture] Warning: No suitable device found: {e}")
            return None

    def _get_device_sample_rate(self) -> int:
        try:
            if self._device_id is not None:
                device_info = sd.query_devices(self._device_id)
            else:
                device_info = sd.query_devices(kind='input')
            return int(device_info['default_samplerate'])
        except Exception:
            return self.device_sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if self._running:
            self._audio_queue.put(indata.copy())

    def _writer_loop(self):
        while self._running or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.1)
                self._process_chunk(data)
            except queue.Empty:
                continue

    def _process_chunk(self, data: np.ndarray):
        """
        Process an audio chunk: resample, keep stereo, add to buffer.

        Args:
            data: Raw audio data from device (samples, channels)
        """
        # data shape: (N, device_channels) — typically (N, 2) for stereo
        if data.ndim == 1:
            # Mono input: duplicate to stereo
            stereo = np.stack([data, data], axis=0)  # (2, N)
        elif data.shape[1] >= 2:
            # Take first 2 channels, transpose to (2, N)
            stereo = data[:, :2].T.astype(np.float32)
        else:
            # Single channel: duplicate
            stereo = np.stack([data[:, 0], data[:, 0]], axis=0)

        # Resample each channel if needed
        if self._actual_sample_rate != self.sample_rate:
            left = self._resample(stereo[0], self._actual_sample_rate, self.sample_rate)
            right = self._resample(stereo[1], self._actual_sample_rate, self.sample_rate)
            stereo = np.stack([left, right], axis=0)

        # Add to rolling stereo buffer
        n = stereo.shape[1]
        with self._buffer_lock:
            self._total_samples += n

            if self._buffer_pos + n <= self.buffer_size:
                self._buffer[:, self._buffer_pos:self._buffer_pos + n] = stereo
                self._buffer_pos += n
            else:
                shift = n
                self._buffer[:, :-shift] = self._buffer[:, shift:].copy()
                self._buffer[:, -n:] = stereo
                self._buffer_pos = self.buffer_size

        # Notify callback if set
        if self._callback:
            self._callback(stereo)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)

        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def set_callback(self, callback: Callable[[np.ndarray], None]):
        self._callback = callback

    def start(self):
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioCapture] sounddevice not available, running without audio")
            return

        if self._running:
            print("[AudioCapture] Already running")
            return

        self._running = True

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        try:
            self._stream = sd.InputStream(
                device=self._device_id,
                channels=2,  # Capture stereo
                samplerate=self._actual_sample_rate,
                dtype='float32',
                blocksize=self.blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            print(f"[AudioCapture] Started stereo capture (device rate: {self._actual_sample_rate} Hz, "
                  f"target: {self.sample_rate} Hz, buffer: {self.buffer_duration}s)")
        except Exception as e:
            print(f"[AudioCapture] Failed to start: {e}")
            self._running = False

    def stop(self):
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
        Get current stereo audio buffer.

        Returns:
            (2, buffer_size) stereo float32, normalized
        """
        with self._buffer_lock:
            buffer = self._buffer.copy()

        # Normalize per-channel
        max_val = np.abs(buffer).max()
        if max_val > 0:
            buffer = buffer / max_val * 0.95

        return buffer

    def get_buffer_filled(self) -> float:
        with self._buffer_lock:
            return self._buffer_pos / self.buffer_size

    @property
    def total_samples(self) -> int:
        with self._buffer_lock:
            return self._total_samples

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_available(self) -> bool:
        return SOUNDDEVICE_AVAILABLE


class DummyAudioCapture:
    """
    Dummy audio capture for when sounddevice is not available.
    Returns zeros for all audio queries.
    """

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 16.0,
                 channels: int = 2, **kwargs):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = int(buffer_duration * sample_rate)
        self._running = False
        self._total_samples = 0

    def start(self):
        self._running = True
        print("[DummyAudioCapture] Running without audio (sounddevice not available)")

    def stop(self):
        self._running = False

    def get_buffer(self) -> np.ndarray:
        return np.zeros((self.channels, self.buffer_size), dtype=np.float32)

    def get_buffer_filled(self) -> float:
        return 0.0

    def set_callback(self, callback):
        pass

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_available(self) -> bool:
        return False


def get_audio_capture(**kwargs) -> AudioCapture:
    """Factory function to get appropriate audio capture."""
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

        name_lower = device['name'].lower()
        if any(k in name_lower for k in ['loopback', 'stereo mix', 'what u hear']):
            print(f"    *** LOOPBACK DEVICE ***")
        print()
