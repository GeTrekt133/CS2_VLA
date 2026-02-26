"""
Async Frame Writer — saves frames to disk using ThreadPoolExecutor.

At 64fps with 640x640 JPEG: ~5ms/frame encode, 8 threads keeps up with margin.
Backpressure: drops oldest frames if queue exceeds max_pending.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict
from collections import deque

import cv2
import numpy as np


class FrameSaver:

    def __init__(
        self,
        num_threads: int = 8,
        jpeg_quality: int = 85,
        frame_size: int = 640,
        max_pending: int = 128,
    ):
        self._num_threads = num_threads
        self._quality = jpeg_quality
        self._frame_size = frame_size
        self._max_pending = max_pending

        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        self._pending: deque[Future] = deque()
        self._lock = threading.Lock()

        self._stats = {
            "saved": 0,
            "dropped": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def save(self, frame: np.ndarray, tick: int, output_dir: str):
        """Submit a frame for async saving as tick_{tick}.jpg."""
        # Cleanup completed futures
        self._cleanup_done()

        # Backpressure: drop if too many pending
        with self._lock:
            if len(self._pending) >= self._max_pending:
                self._stats["dropped"] += 1
                return

        path = os.path.join(output_dir, f"tick_{tick}.jpg")
        # Copy frame to avoid race conditions (WGC might reuse buffer)
        frame_copy = frame.copy()
        future = self._executor.submit(self._save_worker, frame_copy, path)

        with self._lock:
            self._pending.append(future)

    def _save_worker(self, frame: np.ndarray, path: str):
        try:
            # Resize to target size
            if frame.shape[0] != self._frame_size or frame.shape[1] != self._frame_size:
                frame = cv2.resize(frame, (self._frame_size, self._frame_size))

            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])

            with self._lock:
                self._stats["saved"] += 1
        except Exception as e:
            with self._lock:
                self._stats["errors"] += 1
            print(f"[FrameSaver] Error saving {path}: {e}")

    def _cleanup_done(self):
        """Remove completed futures from the pending deque."""
        with self._lock:
            while self._pending and self._pending[0].done():
                self._pending.popleft()

    def flush(self, timeout: float = 30.0):
        """Wait for all pending saves to complete."""
        deadline = time.time() + timeout
        while True:
            self._cleanup_done()
            with self._lock:
                if not self._pending:
                    break
            if time.time() > deadline:
                print(f"[FrameSaver] Flush timeout, {len(self._pending)} frames still pending")
                break
            time.sleep(0.05)

    def shutdown(self):
        """Flush and shutdown the executor."""
        self.flush()
        self._executor.shutdown(wait=True)

    def print_stats(self):
        s = self.stats
        print(f"[FrameSaver] Saved: {s['saved']}, Dropped: {s['dropped']}, Errors: {s['errors']}")
