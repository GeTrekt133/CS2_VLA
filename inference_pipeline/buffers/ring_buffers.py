"""
Thread-safe ring buffers for streaming data (frames, audio, actions).
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

    def __init__(self, max_size: int = 129):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, timestamp: float, frame: np.ndarray):
        """
        Add a frame to the buffer.

        Args:
            timestamp: Frame timestamp (time.time())
            frame: Image frame (H, W, C) or (C, H, W)
        """
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))

    def get_recent(self, n: int) -> List[Tuple[float, np.ndarray]]:
        """
        Get the n most recent frames.

        Args:
            n: Number of frames to retrieve

        Returns:
            List of (timestamp, frame) pairs, oldest first
        """
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        """Get the most recent frame."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_all(self) -> List[Tuple[float, np.ndarray]]:
        """Get all frames in buffer."""
        with self._lock:
            return list(self._buffer)

    def clear(self):
        """Clear all frames."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether buffer has reached max capacity."""
        with self._lock:
            return len(self._buffer) >= self.max_size


class AudioBuffer:
    """
    Thread-safe rolling audio buffer.

    Maintains a fixed-size circular buffer of audio samples.
    """

    def __init__(self, sample_rate: int = 16000, duration: float = 30.0):
        """
        Initialize audio buffer.

        Args:
            sample_rate: Audio sample rate (Hz)
            duration: Buffer duration (seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer_size = int(sample_rate * duration)

        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0
        self._lock = threading.RLock()

    def add(self, audio: np.ndarray):
        """
        Add audio samples to the buffer.

        Args:
            audio: Audio samples (N,) float32
        """
        with self._lock:
            audio = audio.astype(np.float32).flatten()
            n = len(audio)

            if n == 0:
                return

            self._total_samples += n

            if self._write_pos + n <= self.buffer_size:
                # Fits without wrapping
                self._buffer[self._write_pos:self._write_pos + n] = audio
                self._write_pos += n
            else:
                # Need to shift buffer left and append
                shift = n
                self._buffer[:-shift] = self._buffer[shift:].copy()
                self._buffer[-n:] = audio
                self._write_pos = self.buffer_size

    def get_buffer(self) -> np.ndarray:
        """
        Get current audio buffer.

        Returns:
            Audio buffer (buffer_size,) float32
        """
        with self._lock:
            return self._buffer.copy()

    def get_filled_ratio(self) -> float:
        """Get fraction of buffer filled (0-1)."""
        with self._lock:
            return min(1.0, self._write_pos / self.buffer_size)

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_samples = 0

    def __len__(self) -> int:
        with self._lock:
            return self._write_pos


class ActionHistoryBuffer:
    """
    Buffer for tracking predicted action history.

    Stores the last N action predictions for feeding back to the model.
    """

    def __init__(self, max_size: int = 16, mouse_dim: int = 2, keys_dim: int = 20):
        """
        Initialize action history buffer.

        Args:
            max_size: Number of past actions to store
            mouse_dim: Mouse action dimensions (yaw, pitch)
            keys_dim: Number of key actions
        """
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
            (max_size, mouse_dim + keys_dim) action history
        """
        with self._lock:
            return np.concatenate([
                self._mouse_history,
                self._keys_history
            ], axis=1)

    def get_mouse_history(self) -> np.ndarray:
        """Get mouse delta history (max_size, 2)."""
        with self._lock:
            return self._mouse_history.copy()

    def get_keys_history(self) -> np.ndarray:
        """Get keys history (max_size, 20)."""
        with self._lock:
            return self._keys_history.copy()

    def clear(self):
        """Clear history."""
        with self._lock:
            self._mouse_history.fill(0)
            self._keys_history.fill(0)
            self._write_pos = 0

    @property
    def filled_count(self) -> int:
        """Number of actions in history."""
        with self._lock:
            return self._write_pos


class StateBuffer:
    """
    Buffer for game state vectors from GSI.
    """

    def __init__(self, max_size: int = 16, state_dim: int = 95):
        """
        Initialize state buffer.

        Args:
            max_size: Number of states to store
            state_dim: Dimension of state vector
        """
        self.max_size = max_size
        self.state_dim = state_dim

        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, timestamp: float, state_vec: np.ndarray):
        """Add a state vector."""
        with self._lock:
            self._buffer.append((timestamp, state_vec.copy()))

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        """Get the most recent state."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_recent(self, n: int) -> List[np.ndarray]:
        """Get n most recent state vectors."""
        with self._lock:
            states = list(self._buffer)[-n:]
            return [s[1] for s in states]

    def clear(self):
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


def test_buffers():
    """Test buffer implementations."""
    print("Testing FrameBuffer...")
    fb = FrameBuffer(max_size=5)

    for i in range(10):
        frame = np.random.rand(480, 640, 3).astype(np.float32)
        fb.add(time.time(), frame)

    print(f"  Buffer length: {len(fb)} (max: 5)")
    print(f"  Recent 3: {len(fb.get_recent(3))} frames")

    print("\nTesting AudioBuffer...")
    ab = AudioBuffer(sample_rate=16000, duration=1.0)

    for i in range(5):
        chunk = np.random.rand(3200).astype(np.float32)
        ab.add(chunk)

    print(f"  Buffer length: {len(ab)}")
    print(f"  Filled ratio: {ab.get_filled_ratio():.2f}")
    print(f"  Buffer shape: {ab.get_buffer().shape}")

    print("\nTesting ActionHistoryBuffer...")
    ahb = ActionHistoryBuffer(max_size=16)

    for i in range(20):
        mouse = np.random.rand(2).astype(np.float32)
        keys = np.random.rand(20).astype(np.float32)
        ahb.add(mouse, keys)

    print(f"  History shape: {ahb.get_history().shape}")
    print(f"  Filled count: {ahb.filled_count}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_buffers()
