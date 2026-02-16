"""
Screen capture module using mss for fast screenshot acquisition.
Provides both full scene frames and cropped radar images.
"""

import time
import threading
from typing import Optional, Tuple, Callable
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
    Fast screen capture using mss library.

    Captures full screen frames and extracts radar crop in a separate thread.
    Pushes frames to provided callbacks for buffer storage.
    """

    def __init__(
        self,
        screen_width: int = 640,
        screen_height: int = 480,
        radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170),
        radar_size: Tuple[int, int] = (224, 224),
        target_fps: int = 60,
        monitor_index: int = 1,
    ):
        """
        Initialize screen capture.

        Args:
            screen_width: Target width for scene frames
            screen_height: Target height for scene frames
            radar_crop_box: (left, top, right, bottom) for radar crop
            radar_size: Target size for radar after resize
            target_fps: Target capture rate
            monitor_index: Monitor to capture (1 = primary)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radar_crop_box = radar_crop_box
        self.radar_size = radar_size
        self.target_fps = target_fps
        self.monitor_index = monitor_index

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sct: Optional[mss.mss] = None

        # Callbacks for frame delivery
        self._scene_callback: Optional[Callable[[float, np.ndarray], None]] = None
        self._radar_callback: Optional[Callable[[float, np.ndarray], None]] = None

        # Stats
        self._frame_count = 0
        self._last_fps_time = 0.0
        self._current_fps = 0.0

    def set_scene_callback(self, callback: Callable[[float, np.ndarray], None]):
        """Set callback for scene frames: callback(timestamp, frame)."""
        self._scene_callback = callback

    def set_radar_callback(self, callback: Callable[[float, np.ndarray], None]):
        """Set callback for radar frames: callback(timestamp, frame)."""
        self._radar_callback = callback

    def start(self):
        """Start capture thread."""
        if not MSS_AVAILABLE:
            raise RuntimeError("mss library not available")

        if self._running:
            print("[ScreenCapture] Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[ScreenCapture] Started at {self.target_fps} FPS target")

    def stop(self):
        """Stop capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print(f"[ScreenCapture] Stopped. Total frames: {self._frame_count}")

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        self._sct = mss.mss()
        monitor = self._sct.monitors[self.monitor_index]
        target_interval = 1.0 / self.target_fps

        self._last_fps_time = time.time()
        fps_frame_count = 0

        while self._running:
            try:
                loop_start = time.time()

                # Capture screenshot
                screenshot = self._sct.grab(monitor)
                timestamp = time.time()

                # Convert to numpy array (BGRA -> RGB)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

                # Resize to target dimensions
                scene_frame = cv2.resize(frame, (self.screen_width, self.screen_height))

                # Deliver scene frame
                if self._scene_callback:
                    self._scene_callback(timestamp, scene_frame)

                # Extract and process radar
                if self._radar_callback:
                    radar_frame = self._extract_radar(scene_frame)
                    self._radar_callback(timestamp, radar_frame)

                self._frame_count += 1
                fps_frame_count += 1

                # Update FPS calculation
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._current_fps = fps_frame_count / (now - self._last_fps_time)
                    fps_frame_count = 0
                    self._last_fps_time = now

                # Sleep to maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"[ScreenCapture] Error: {e}")
                time.sleep(0.1)

        # Cleanup
        if self._sct:
            self._sct.close()
            self._sct = None

    def _extract_radar(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract and resize radar from scene frame.

        Args:
            frame: Full scene frame (H, W, 3)

        Returns:
            Radar crop resized to radar_size (224, 224, 3)
        """
        left, top, right, bottom = self.radar_crop_box

        # Crop radar region
        radar = frame[top:bottom, left:right, :]

        # Resize to target size
        radar = cv2.resize(radar, self.radar_size)

        return radar

    def capture_single(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture a single frame (for testing/debugging).

        Returns:
            (scene_frame, radar_frame)
        """
        if not MSS_AVAILABLE:
            raise RuntimeError("mss library not available")

        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            scene = cv2.resize(frame, (self.screen_width, self.screen_height))
            radar = self._extract_radar(scene)

            return scene, radar

    @property
    def fps(self) -> float:
        """Current capture FPS."""
        return self._current_fps

    @property
    def frame_count(self) -> int:
        """Total frames captured."""
        return self._frame_count

    @property
    def is_running(self) -> bool:
        """Whether capture is active."""
        return self._running


def test_screen_capture():
    """Test screen capture functionality."""
    print("Testing ScreenCapture...")

    capture = ScreenCapture()

    # Test single capture
    print("\nTesting single capture...")
    try:
        scene, radar = capture.capture_single()
        print(f"Scene shape: {scene.shape}")
        print(f"Radar shape: {radar.shape}")
        print(f"Scene dtype: {scene.dtype}, range: [{scene.min()}, {scene.max()}]")
    except Exception as e:
        print(f"Single capture failed: {e}")
        return

    # Test continuous capture
    print("\nTesting continuous capture for 3 seconds...")
    frames_received = {"scene": 0, "radar": 0}

    def on_scene(ts, frame):
        frames_received["scene"] += 1

    def on_radar(ts, frame):
        frames_received["radar"] += 1

    capture.set_scene_callback(on_scene)
    capture.set_radar_callback(on_radar)

    capture.start()
    time.sleep(3.0)
    capture.stop()

    print(f"Scene frames: {frames_received['scene']}")
    print(f"Radar frames: {frames_received['radar']}")
    print(f"Average FPS: {frames_received['scene'] / 3.0:.1f}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_screen_capture()
