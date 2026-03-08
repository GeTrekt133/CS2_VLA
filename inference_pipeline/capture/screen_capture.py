"""
Screen capture module using mss for fast screenshot acquisition.
Synchronous grab() model — caller controls the loop rate.
"""

import time
from typing import Tuple
import numpy as np
import cv2

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[ScreenCapture] WARNING: mss not installed. pip install mss")


class ScreenCapture:
    """
    Synchronous screen capture using mss.

    Usage:
        cap = ScreenCapture(screen_width=640, screen_height=640)
        cap.open()
        ts, scene, radar = cap.grab()
        ...
        cap.close()
    """

    def __init__(
        self,
        screen_width: int = 640,
        screen_height: int = 640,
        radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170),
        radar_size: Tuple[int, int] = (224, 224),
        monitor_index: int = 1,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radar_crop_box = radar_crop_box
        self.radar_size = radar_size
        self.monitor_index = monitor_index

        self._sct = None
        self._monitor = None
        self._frame_count = 0
        self._fps_frames = 0
        self._fps_start = 0.0
        self._current_fps = 0.0

    def open(self):
        """Initialize mss context."""
        if not MSS_AVAILABLE:
            raise RuntimeError("mss library not available. pip install mss")
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[self.monitor_index]
        self._fps_start = time.time()
        print(f"[ScreenCapture] Opened (monitor {self.monitor_index}, "
              f"{self.screen_width}x{self.screen_height})")

    def close(self):
        """Release mss context."""
        if self._sct:
            self._sct.close()
            self._sct = None
        print(f"[ScreenCapture] Closed. Total frames: {self._frame_count}")

    def grab(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Grab one screenshot synchronously.

        Returns:
            (timestamp, scene_frame, radar_frame)
            scene_frame: (H, W, 3) uint8 RGB, resized to screen_width x screen_height
            radar_frame: (224, 224, 3) uint8 RGB, cropped and resized
        """
        if self._sct is None:
            self.open()

        screenshot = self._sct.grab(self._monitor)
        timestamp = time.time()

        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        scene_frame = cv2.resize(frame, (self.screen_width, self.screen_height))

        # Radar crop
        left, top, right, bottom = self.radar_crop_box
        radar = scene_frame[top:bottom, left:right, :]
        radar_frame = cv2.resize(radar, self.radar_size)

        # FPS tracking
        self._frame_count += 1
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_start = now

        return timestamp, scene_frame, radar_frame

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count


def test_screen_capture():
    """Test screen capture functionality."""
    print("Testing ScreenCapture...")

    cap = ScreenCapture()
    cap.open()

    # Single grab
    ts, scene, radar = cap.grab()
    print(f"Scene: {scene.shape} {scene.dtype}, Radar: {radar.shape}")

    # Speed test (3 seconds)
    print("\nSpeed test (3 seconds)...")
    start = time.time()
    count = 0
    while time.time() - start < 3.0:
        cap.grab()
        count += 1
    elapsed = time.time() - start
    print(f"Frames: {count}, FPS: {count / elapsed:.1f}")

    cap.close()
    print("Test complete!")


if __name__ == "__main__":
    test_screen_capture()
