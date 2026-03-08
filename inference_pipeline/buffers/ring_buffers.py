"""
Thread-safe ring buffers for streaming data (frames, audio, actions).

Aligned with final_model/ architecture:
  - StereoAudioBuffer: (2, N) stereo, 16 sec @ 16kHz
  - ActionHistoryBuffer: max_size=64 (TemporalTransformer expects 64 action tokens)
  - StateBuffer: state_dim=100
"""

import time
import threading
from typing import Optional, List, Tuple, Deque
from collections import deque
import numpy as np


class FrameBuffer:
    """
    Thread-safe circular buffer for image frames.

    Stores (timestamp, frame) pairs with automatic eviction of old frames.
    """

    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, timestamp: float, frame: np.ndarray):
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))

    def get_recent(self, n: int) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_all(self) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.max_size


class StereoAudioBuffer:
    """
    Thread-safe rolling stereo audio buffer.

    Stores (2, N) stereo samples for StereoAudioEncoder.
    16 sec @ 16kHz = (2, 256000) samples.
    """

    def __init__(self, sample_rate: int = 16000, duration: float = 16.0, channels: int = 2):
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.buffer_size = int(sample_rate * duration)  # samples per channel

        self._buffer = np.zeros((channels, self.buffer_size), dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0
        self._lock = threading.RLock()

    def add(self, audio: np.ndarray):
        """
        Add stereo audio samples to the buffer.

        Args:
            audio: (2, N) stereo or (N, 2) interleaved stereo float32
        """
        with self._lock:
            # Normalize to (2, N)
            if audio.ndim == 1:
                # Mono fallback: duplicate to both channels
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim == 2:
                if audio.shape[0] != self.channels and audio.shape[1] == self.channels:
                    audio = audio.T  # (N, 2) → (2, N)
                elif audio.shape[0] != self.channels:
                    # Take first 2 channels or duplicate
                    if audio.shape[0] > self.channels:
                        audio = audio[:self.channels]
                    else:
                        audio = np.stack([audio[0], audio[0]], axis=0)

            audio = audio.astype(np.float32)
            n = audio.shape[1]

            if n == 0:
                return

            self._total_samples += n

            if self._write_pos + n <= self.buffer_size:
                self._buffer[:, self._write_pos:self._write_pos + n] = audio
                self._write_pos += n
            else:
                # Shift buffer left and append
                shift = n
                self._buffer[:, :-shift] = self._buffer[:, shift:].copy()
                self._buffer[:, -n:] = audio
                self._write_pos = self.buffer_size

    def get_buffer(self) -> np.ndarray:
        """
        Get current stereo audio buffer.

        Returns:
            (2, buffer_size) float32 stereo audio
        """
        with self._lock:
            return self._buffer.copy()

    def get_filled_ratio(self) -> float:
        with self._lock:
            return min(1.0, self._write_pos / self.buffer_size)

    def clear(self):
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_samples = 0

    def __len__(self) -> int:
        with self._lock:
            return self._write_pos


# Keep old AudioBuffer for backwards compatibility
class AudioBuffer(StereoAudioBuffer):
    """Alias for StereoAudioBuffer."""
    pass


class ActionHistoryBuffer:
    """
    Buffer for tracking predicted action history.

    TemporalCrossTransformer expects (B, 64, 22) action input which gets compressed
    64→16 via ModalityCompressor. So we keep 64 action history entries.
    """

    def __init__(self, max_size: int = 64, mouse_dim: int = 2, keys_dim: int = 20):
        self.max_size = max_size
        self.mouse_dim = mouse_dim
        self.keys_dim = keys_dim

        self._mouse_history = np.zeros((max_size, mouse_dim), dtype=np.float32)
        self._keys_history = np.zeros((max_size, keys_dim), dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.RLock()

    def add(self, mouse_delta: np.ndarray, key_probs: np.ndarray):
        """
        Add a new action to history.

        Args:
            mouse_delta: (2,) yaw/pitch delta
            key_probs: (20,) key probabilities (after sigmoid)
        """
        with self._lock:
            # Shift history left
            self._mouse_history[:-1] = self._mouse_history[1:]
            self._keys_history[:-1] = self._keys_history[1:]

            # Add new action at the end
            self._mouse_history[-1] = mouse_delta[:self.mouse_dim]
            self._keys_history[-1] = (key_probs[:self.keys_dim] > 0.5).astype(np.float32)

            self._write_pos = min(self._write_pos + 1, self.max_size)

    def get_history(self) -> np.ndarray:
        """
        Get combined action history for model input.

        Returns:
            (max_size, mouse_dim + keys_dim) = (64, 22)
        """
        with self._lock:
            return np.concatenate([
                self._mouse_history,
                self._keys_history
            ], axis=1)

    def get_mouse_history(self) -> np.ndarray:
        with self._lock:
            return self._mouse_history.copy()

    def get_keys_history(self) -> np.ndarray:
        with self._lock:
            return self._keys_history.copy()

    def clear(self):
        with self._lock:
            self._mouse_history.fill(0)
            self._keys_history.fill(0)
            self._write_pos = 0

    @property
    def filled_count(self) -> int:
        with self._lock:
            return self._write_pos


class StateBuffer:
    """Buffer for game state vectors from GSI. State dim = 100."""

    def __init__(self, max_size: int = 16, state_dim: int = 100):
        self.max_size = max_size
        self.state_dim = state_dim

        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, timestamp: float, state_vec: np.ndarray):
        with self._lock:
            self._buffer.append((timestamp, state_vec.copy()))

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_recent(self, n: int) -> List[np.ndarray]:
        with self._lock:
            states = list(self._buffer)[-n:]
            return [s[1] for s in states]

    def clear(self):
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class DetectionBuffer:
    """
    Ring buffer for detection vectors.

    Each detection vector is (detection_dim,) = (100,) = 20 dets × 5 features.
    TemporalCrossTransformer expects (B, 16, 100) detection sequence.
    """

    def __init__(self, max_size: int = 16, detection_dim: int = 100):
        self.max_size = max_size
        self.detection_dim = detection_dim

        self._buffer = np.zeros((max_size, detection_dim), dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.RLock()

    def add(self, detection_vec: np.ndarray):
        """Add detection vector (100,)."""
        with self._lock:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = detection_vec[:self.detection_dim]
            self._write_pos = min(self._write_pos + 1, self.max_size)

    def get_sequence(self) -> np.ndarray:
        """Get full detection sequence (max_size, detection_dim) = (16, 100)."""
        with self._lock:
            return self._buffer.copy()

    def clear(self):
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0

    @property
    def filled_count(self) -> int:
        with self._lock:
            return self._write_pos
