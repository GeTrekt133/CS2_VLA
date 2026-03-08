"""
Audio Recorder — captures system audio via WASAPI loopback.

Uses pyaudiowpatch (WASAPI loopback) for capture + ffmpeg for resampling.
No Stereo Mix setup needed — captures directly from the output device.

Prerequisites:
  - pip install PyAudioWPatch
  - ffmpeg in PATH (or specify ffmpeg_path) — for resampling only
"""

import os
import shutil
import struct
import subprocess
import threading
import time
import wave
from typing import Optional

# Common ffmpeg locations on Windows
_FFMPEG_SEARCH_PATHS = [
    os.path.expanduser("~/Downloads/ffmpeg-8.0.1-essentials_build/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe"),
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]


def _find_ffmpeg(explicit_path: Optional[str] = None) -> str:
    """Find ffmpeg binary. Returns path or 'ffmpeg' (hope it's in PATH)."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    # Check PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Check common locations
    for p in _FFMPEG_SEARCH_PATHS:
        if os.path.isfile(p):
            return p

    return "ffmpeg"  # fallback, will fail with clear error


def _find_loopback_device():
    """Find WASAPI loopback device for the default audio output.

    Returns (device_info_dict, pyaudio_instance) or (None, None).
    """
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if (dev.get("isLoopbackDevice", False) and
                    dev["name"].startswith(default_speakers["name"])):
                return dev, p

        p.terminate()
        return None, None

    except ImportError:
        print("[AudioRecorder] pyaudiowpatch not installed: pip install PyAudioWPatch")
        return None, None
    except Exception as e:
        print(f"[AudioRecorder] WASAPI loopback init failed: {e}")
        return None, None


class AudioRecorder:
    """Records system audio via WASAPI loopback (pyaudiowpatch).

    Captures at native rate (48kHz stereo) then resamples to target rate
    (16kHz mono) using ffmpeg.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device_name: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self._ffmpeg = _find_ffmpeg(ffmpeg_path)
        self._output_path: Optional[str] = None
        self._recording = False

        # WASAPI loopback state
        self._pyaudio = None
        self._stream = None
        self._wav_file = None
        self._temp_path: Optional[str] = None
        self._loopback_device = None

        # Find loopback device
        self._loopback_device, self._pyaudio = _find_loopback_device()

        if self._loopback_device:
            name = self._loopback_device["name"]
            rate = int(self._loopback_device["defaultSampleRate"])
            ch = self._loopback_device["maxInputChannels"]
            print(f"[AudioRecorder] WASAPI loopback: {name} ({rate}Hz, {ch}ch)")
        else:
            print("[AudioRecorder] WARNING: No WASAPI loopback device found!")

    def start_recording(self, output_path: str):
        """Start recording audio via WASAPI loopback."""
        if self._recording:
            print("[AudioRecorder] Already recording!")
            return

        if not self._loopback_device:
            print("[AudioRecorder] No loopback device, skipping")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._output_path = output_path
        self._temp_path = output_path + ".tmp.wav"

        dev = self._loopback_device
        native_rate = int(dev["defaultSampleRate"])
        channels = dev["maxInputChannels"]

        import pyaudiowpatch as pyaudio

        # Open temp WAV for raw capture (native rate, stereo, float32)
        self._wav_file = wave.open(self._temp_path, "wb")
        self._wav_file.setnchannels(channels)
        self._wav_file.setsampwidth(4)  # float32 = 4 bytes
        self._wav_file.setframerate(native_rate)
        # Set WAVE_FORMAT_IEEE_FLOAT (3) instead of PCM (1)
        # wave module writes PCM by default, we need to fix the header
        # Actually, wave module only supports PCM. Write as int16 instead.
        self._wav_file.close()

        # Reopen with int16 (wave module supports this natively)
        self._wav_file = wave.open(self._temp_path, "wb")
        self._wav_file.setnchannels(channels)
        self._wav_file.setsampwidth(2)  # int16
        self._wav_file.setframerate(native_rate)

        def callback(in_data, frame_count, time_info, status):
            if self._wav_file and self._recording:
                self._wav_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=512,
            stream_callback=callback,
        )
        self._recording = True
        print(f"[AudioRecorder] Recording -> {output_path}")

    def stop_recording(self) -> Optional[str]:
        """Stop recording, resample to target rate, return output path."""
        if not self._recording:
            return None

        self._recording = False

        # Stop stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Close temp WAV
        if self._wav_file:
            try:
                self._wav_file.close()
            except Exception:
                pass
            self._wav_file = None

        # Resample temp WAV -> target format (16kHz mono float32)
        if self._temp_path and os.path.exists(self._temp_path):
            temp_size = os.path.getsize(self._temp_path)
            if temp_size < 100:
                print(f"[AudioRecorder] WARNING: Temp file too small ({temp_size} bytes)")
                self._cleanup_temp()
                return None

            try:
                result = subprocess.run(
                    [
                        self._ffmpeg,
                        "-i", self._temp_path,
                        "-ac", "2",                     # stereo
                        "-ar", str(self.sample_rate),    # resample
                        "-acodec", "pcm_f32le",         # float32 WAV
                        "-y",                            # overwrite
                        self._output_path,
                    ],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    print(f"[AudioRecorder] ffmpeg resample failed (exit={result.returncode})")
                    self._cleanup_temp()
                    return None
            except Exception as e:
                print(f"[AudioRecorder] ffmpeg error: {e}")
                self._cleanup_temp()
                return None

            self._cleanup_temp()

            if os.path.exists(self._output_path):
                duration = self._get_duration(self._output_path)
                print(f"[AudioRecorder] Saved: {self._output_path} ({duration:.1f}s)")
                return self._output_path

        return None

    def _cleanup_temp(self):
        """Remove temporary capture file."""
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass

    def _get_duration(self, path: str) -> float:
        """Get WAV duration in seconds."""
        try:
            import soundfile as sf
            info = sf.info(path)
            return info.duration
        except Exception:
            try:
                size = os.path.getsize(path)
                # float32 stereo WAV: 4 bytes per sample × 2 channels
                return (size - 44) / (4 * 2 * self.sample_rate)
            except Exception:
                return 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def cleanup(self):
        """Release pyaudio resources."""
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None


def list_audio_devices(ffmpeg_path: Optional[str] = None):
    """List WASAPI loopback devices."""
    print("\n=== Audio Devices (WASAPI loopback) ===\n")

    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        print(f"  Default output: {default_speakers['name']}")
        print(f"    Rate: {int(default_speakers['defaultSampleRate'])}Hz, "
              f"Channels: {default_speakers['maxOutputChannels']}\n")

        print("  Loopback devices:")
        found = False
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                is_default = dev["name"].startswith(default_speakers["name"])
                tag = " <-- ACTIVE (will be used)" if is_default else ""
                print(f"    [{i}] {dev['name']} "
                      f"({int(dev['defaultSampleRate'])}Hz, {dev['maxInputChannels']}ch)"
                      f"{tag}")
                found = True

        if not found:
            print("    No loopback devices found!")

        p.terminate()

        # Also show ffmpeg info
        ffmpeg = _find_ffmpeg(ffmpeg_path)
        print(f"\n  ffmpeg (for resampling): {ffmpeg}")

    except ImportError:
        print("  pyaudiowpatch not installed!")
        print("  Install: pip install PyAudioWPatch")
    except Exception as e:
        print(f"  Error: {e}")
