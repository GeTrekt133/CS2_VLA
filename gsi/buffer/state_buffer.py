"""
Circular buffers for game states and frames.

Provides thread-safe storage for:
- Game state history (for model input sequences)
- Screenshot frames (synchronized with states)
"""
import numpy as np
import threading
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass

from ..adapter.state_adapter import GameStateSnapshot


@dataclass
class BufferedState:
    """State with metadata for buffering."""
    snapshot: GameStateSnapshot
    state_vec: np.ndarray
    timestamp: float
    frame_path: Optional[str] = None
    frame: Optional[np.ndarray] = None


class StateBuffer:
    """
    Thread-safe circular buffer for game states.

    Maintains history for:
    - state_vec: (buffer_size, 95)
    - snapshots with full game info
    - timestamps for synchronization

    Usage:
        buffer = StateBuffer(size=256)
        buffer.add(snapshot, state_vec)

        # Get recent history for inference
        states, timestamps = buffer.get_recent(n=16)
    """

    def __init__(self, size: int = 256):
        self.size = size
        self._buffer: deque = deque(maxlen=size)
        self._lock = threading.RLock()

    def add(
        self,
        snapshot: GameStateSnapshot,
        state_vec: np.ndarray,
        frame_path: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
    ):
        """Add a new state to the buffer."""
        with self._lock:
            buffered = BufferedState(
                snapshot=snapshot,
                state_vec=state_vec.copy(),
                timestamp=snapshot.timestamp,
                frame_path=frame_path,
                frame=frame,
            )
            self._buffer.append(buffered)

    def get_recent(self, n: int) -> Tuple[np.ndarray, List[float]]:
        """
        Get the n most recent states.

        Returns:
            state_vecs: np.ndarray (n, 95) or less if buffer not full
            timestamps: list of timestamps
        """
        with self._lock:
            items = list(self._buffer)[-n:]

            if not items:
                return np.zeros((0, 95), dtype=np.float32), []

            state_vecs = np.stack([item.state_vec for item in items])
            timestamps = [item.timestamp for item in items]

            return state_vecs, timestamps

    def get_recent_snapshots(self, n: int) -> List[BufferedState]:
        """Get n most recent buffered states with full data."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_state_at_time(self, target_time: float) -> Optional[BufferedState]:
        """Get the state closest to target timestamp."""
        with self._lock:
            if not self._buffer:
                return None

            closest = min(
                self._buffer,
                key=lambda x: abs(x.timestamp - target_time)
            )
            return closest

    def get_current(self) -> Optional[BufferedState]:
        """Get the most recent state."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def get_all(self) -> List[BufferedState]:
        """Get all states in buffer."""
        with self._lock:
            return list(self._buffer)

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self._buffer)


class FrameBuffer:
    """
    Buffer for synchronized screenshots.

    Stores frames with timestamps for matching with GSI states.
    """

    def __init__(self, size: int = 64):
        self.size = size
        self._buffer: deque = deque(maxlen=size)  # (timestamp, frame)
        self._lock = threading.RLock()

    def add(self, timestamp: float, frame: np.ndarray):
        """Add a frame with timestamp."""
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))

    def get_frames_in_range(
        self,
        start_time: float,
        end_time: float,
        step: int = 1
    ) -> List[Tuple[float, np.ndarray]]:
        """Get frames within time range."""
        with self._lock:
            frames = [
                (ts, frame) for ts, frame in self._buffer
                if start_time <= ts <= end_time
            ]
            return frames[::step]

    def get_recent(self, n: int) -> List[Tuple[float, np.ndarray]]:
        """Get n most recent frames."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_closest(self, target_time: float, max_diff: float = 0.5) -> Optional[Tuple[float, np.ndarray]]:
        """
        Get frame closest to target timestamp.

        Args:
            target_time: Target timestamp
            max_diff: Maximum allowed time difference (seconds)

        Returns:
            (timestamp, frame) or None if no frame within max_diff
        """
        with self._lock:
            if not self._buffer:
                return None

            closest = min(self._buffer, key=lambda x: abs(x[0] - target_time))

            if abs(closest[0] - target_time) <= max_diff:
                return closest
            return None

    def get_current(self) -> Optional[Tuple[float, np.ndarray]]:
        """Get most recent frame."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self._buffer)
