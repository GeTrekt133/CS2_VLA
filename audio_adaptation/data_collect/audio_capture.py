"""
Audio Capture for CS2 Demo Recording.

Captures system audio (game sound) using WASAPI loopback on Windows.
Can be used standalone or integrated with screen_capture.py.

Usage:
    Standalone:
        python audio_capture.py --output ./audio --duration 60

    Integrated (in screen_capture.py):
        from audio_capture import AudioRecorder
        recorder = AudioRecorder(output_dir="./audio")
        recorder.start_recording("round_1.wav")
        # ... game plays ...
        recorder.stop_recording()

Requirements:
    pip install sounddevice soundfile numpy
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import os
import argparse
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000           # Target sample rate (for model)
    channels: int = 1                   # Mono
    dtype: str = 'float32'
    blocksize: int = 1024               # Samples per callback
    device_sample_rate: int = 48000     # Typical WASAPI loopback rate
    buffer_duration: float = 0.5        # Buffer flush interval


class AudioRecorder:
    """
    Real-time audio recorder using WASAPI loopback.

    Captures system audio (including game sound) and saves to WAV files.
    Automatically resamples from device rate to target rate (16kHz).
    """

    def __init__(
        self,
        output_dir: str = "./audio",
        config: Optional[AudioConfig] = None
    ):
        self.output_dir = output_dir
        self.config = config or AudioConfig()

        # Recording state
        self._recording = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._audio_data: List[np.ndarray] = []
        self._current_file: Optional[str] = None
        self._stream: Optional[sd.InputStream] = None
        self._writer_thread: Optional[threading.Thread] = None

        # Find loopback device
        self._device_id = self._find_loopback_device()
        self._actual_sample_rate = self._get_device_sample_rate()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"[AudioRecorder] Initialized")
        print(f"  Device: {self._device_id}")
        print(f"  Device sample rate: {self._actual_sample_rate} Hz")
        print(f"  Target sample rate: {self.config.sample_rate} Hz")
        print(f"  Output dir: {output_dir}")

    def _find_loopback_device(self) -> Optional[int]:
        """Find WASAPI loopback device for system audio capture."""
        devices = sd.query_devices()

        # Look for loopback device (Windows WASAPI)
        loopback_keywords = ['loopback', 'stereo mix', 'what u hear', 'wave out']

        for i, device in enumerate(devices):
            name_lower = device['name'].lower()
            if any(keyword in name_lower for keyword in loopback_keywords):
                if device['max_input_channels'] > 0:
                    print(f"[AudioRecorder] Found loopback device: {device['name']}")
                    return i

        # Try to use default input as fallback
        try:
            default_input = sd.query_devices(kind='input')
            print(f"[AudioRecorder] Using default input: {default_input['name']}")
            return None  # None means default device
        except Exception as e:
            print(f"[AudioRecorder] Warning: No suitable device found: {e}")
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
            return self.config.device_sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"[AudioRecorder] Stream status: {status}")

        if self._recording:
            # Copy data to queue
            self._audio_queue.put(indata.copy())

    def _writer_loop(self):
        """Background thread for processing and saving audio."""
        while self._recording or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.1)
                self._audio_data.append(data)
            except queue.Empty:
                continue

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        # Calculate ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        # Use numpy interp for simple resampling
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)

        if audio.ndim == 1:
            return np.interp(new_indices, old_indices, audio).astype(np.float32)
        else:
            # Multi-channel
            resampled = np.zeros((new_length, audio.shape[1]), dtype=np.float32)
            for ch in range(audio.shape[1]):
                resampled[:, ch] = np.interp(new_indices, old_indices, audio[:, ch])
            return resampled

    def start_recording(self, filename: str = "recording.wav"):
        """Start recording audio."""
        if self._recording:
            print("[AudioRecorder] Already recording!")
            return

        self._current_file = os.path.join(self.output_dir, filename)
        self._audio_data = []
        self._recording = True

        # Start writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._device_id,
                channels=2,  # Capture stereo, convert to mono later
                samplerate=self._actual_sample_rate,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            print(f"[AudioRecorder] Started recording: {filename}")
        except Exception as e:
            print(f"[AudioRecorder] Failed to start recording: {e}")
            self._recording = False

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to file."""
        if not self._recording:
            print("[AudioRecorder] Not recording!")
            return None

        self._recording = False

        # Stop stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Wait for writer thread
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None

        # Process remaining queue
        while not self._audio_queue.empty():
            try:
                data = self._audio_queue.get_nowait()
                self._audio_data.append(data)
            except queue.Empty:
                break

        if not self._audio_data:
            print("[AudioRecorder] No audio data captured!")
            return None

        # Concatenate all chunks
        audio = np.concatenate(self._audio_data, axis=0)

        # Convert stereo to mono
        if audio.ndim > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)

        # Resample to target rate
        if self._actual_sample_rate != self.config.sample_rate:
            audio = self._resample(audio, self._actual_sample_rate, self.config.sample_rate)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Save to file
        try:
            sf.write(
                self._current_file,
                audio,
                self.config.sample_rate,
                subtype='FLOAT'
            )
            duration = len(audio) / self.config.sample_rate
            print(f"[AudioRecorder] Saved: {self._current_file} ({duration:.1f}s)")
            return self._current_file
        except Exception as e:
            print(f"[AudioRecorder] Failed to save: {e}")
            return None

    def record_duration(self, filename: str, duration: float):
        """Record for a specific duration."""
        self.start_recording(filename)
        time.sleep(duration)
        return self.stop_recording()


class SyncedAudioVideoRecorder:
    """
    Synchronized audio and video recorder for CS2 demos.

    Coordinates audio recording with OBS video recording.
    """

    def __init__(
        self,
        output_dir: str,
        audio_config: Optional[AudioConfig] = None
    ):
        self.output_dir = output_dir
        self.audio_recorder = AudioRecorder(
            output_dir=os.path.join(output_dir, "audio"),
            config=audio_config
        )
        self._sync_offset: float = 0.0  # Audio-video sync offset

    def start_round_recording(self, round_id: int):
        """Start recording audio for a round."""
        filename = f"round_{round_id}.wav"
        self.audio_recorder.start_recording(filename)

    def stop_round_recording(self) -> Optional[str]:
        """Stop round recording and return audio path."""
        return self.audio_recorder.stop_recording()

    def record_round(self, round_id: int, duration: float) -> Optional[str]:
        """Record a full round."""
        filename = f"round_{round_id}.wav"
        return self.audio_recorder.record_duration(filename, duration)


def list_audio_devices():
    """List all available audio devices."""
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


def test_recording():
    """Test audio recording."""
    print("\n=== Audio Recording Test ===\n")

    recorder = AudioRecorder(output_dir="./test_audio")

    print("Recording for 5 seconds...")
    audio_file = recorder.record_duration("test.wav", 5.0)

    if audio_file:
        print(f"\nTest recording saved to: {audio_file}")
        print("Play it back to verify audio capture is working.")
    else:
        print("\nRecording failed!")


def main():
    parser = argparse.ArgumentParser(description="CS2 Audio Capture Tool")
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    parser.add_argument('--test', action='store_true', help='Test recording')
    parser.add_argument('--output', type=str, default='./audio', help='Output directory')
    parser.add_argument('--duration', type=float, default=10.0, help='Recording duration')
    parser.add_argument('--filename', type=str, default='recording.wav', help='Output filename')

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.test:
        test_recording()
        return

    # Default: record for specified duration
    recorder = AudioRecorder(output_dir=args.output)
    print(f"Recording for {args.duration} seconds...")
    audio_file = recorder.record_duration(args.filename, args.duration)

    if audio_file:
        print(f"Recording saved to: {audio_file}")


if __name__ == "__main__":
    main()
