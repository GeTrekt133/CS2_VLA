"""
Screen Capture — captures CS2 window frames using Windows Graphics Capture API.

Uses `windows-capture` library (WGC backend) which works even when the
CS2 window is behind other windows (occluded). Does NOT work if minimized.

Fallback: DxgiDuplicationSession for visible-window capture (faster but
requires window to be visible).

Install: pip install windows-capture
"""

import threading
import time
from typing import Optional

import numpy as np


def _get_client_crop(window_name: str) -> Optional[tuple]:
    """
    Compute the client area crop rect (x1, y1, x2, y2) within the captured window frame.

    WGC captures the full window including title bar and borders.
    This returns the offsets to crop to just the game client area.
    """
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd:
            return None

        # GetWindowRect: absolute screen coords of full window
        wr = win32gui.GetWindowRect(hwnd)     # (left, top, right, bottom)
        # ClientToScreen: client origin (top-left of game area) in screen coords
        pt = win32gui.ClientToScreen(hwnd, (0, 0))
        # GetClientRect: client size (always starts at 0,0)
        cr = win32gui.GetClientRect(hwnd)     # (0, 0, width, height)

        x1 = pt[0] - wr[0]   # pixels from window left to client left
        y1 = pt[1] - wr[1]   # pixels from window top to client top (title bar height)
        x2 = x1 + cr[2]      # x1 + client width
        y2 = y1 + cr[3]      # y1 + client height

        print(f"[WGCCapture] Window {wr[2]-wr[0]}x{wr[3]-wr[1]}, "
              f"client {cr[2]}x{cr[3]}, "
              f"chrome offset top={y1}px left={x1}px")
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"[WGCCapture] Could not compute client crop: {e}")
        return None


class WGCCapture:
    """
    Windows Graphics Capture — per-window capture that works when occluded.

    Stores the latest frame in a thread-safe slot. The capture runs on a
    background thread and updates the slot via on_frame_arrived callback.
    """

    def __init__(self, window_name: str = "Counter-Strike 2"):
        self._window_name = window_name
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_control = None
        self._frame_count = 0
        self._last_frame_time = 0.0  # perf_counter when last frame arrived
        self._last_returned_count = -1  # frame_count at last get_frame() call
        self._client_crop: Optional[tuple] = None  # (x1, y1, x2, y2) to crop title bar

    def start(self):
        """Start capturing in a background thread."""
        if self._running:
            return

        # Compute client area crop (removes title bar and window borders)
        self._client_crop = _get_client_crop(self._window_name)

        self._running = True
        self._frame_count = 0
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()
        # Wait for first frame
        deadline = time.time() + 10.0
        while self._frame_count == 0 and time.time() < deadline:
            time.sleep(0.05)

        if self._frame_count == 0:
            print("[WGCCapture] WARNING: No frames received in 10s. Check window name.")
        else:
            print(f"[WGCCapture] Started, first frame received.")

    def _capture_loop(self):
        """Run WindowsCapture.start() which blocks until stopped."""
        try:
            from windows_capture import WindowsCapture, Frame, InternalCaptureControl

            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=None,
                window_name=self._window_name,
            )

            @capture.event
            def on_frame_arrived(frame: Frame, control: InternalCaptureControl):
                if not self._running:
                    control.stop()
                    return

                # frame.frame_buffer is numpy array (H, W, 4) BGRA uint8
                bgr = frame.frame_buffer[:, :, :3]  # BGRA → BGR (view, no copy yet)

                # Crop out title bar and window borders to get only the client area
                if self._client_crop is not None:
                    x1, y1, x2, y2 = self._client_crop
                    h, w = bgr.shape[:2]
                    x2 = min(x2, w)
                    y2 = min(y2, h)
                    bgr = bgr[y1:y2, x1:x2]

                bgr = bgr.copy()

                with self._lock:
                    self._latest_frame = bgr
                    self._frame_count += 1
                    self._last_frame_time = time.perf_counter()

                # Store control for external stop
                self._capture_control = control

            @capture.event
            def on_closed():
                self._running = False

            capture.start()

        except ImportError:
            print("[WGCCapture] ERROR: windows-capture not installed.")
            print("  Install: pip install windows-capture")
            self._running = False
        except Exception as e:
            print(f"[WGCCapture] ERROR: {e}")
            self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest captured BGR frame, or None if unavailable."""
        with self._lock:
            return self._latest_frame

    def is_stale(self, timeout: float = 2.0) -> bool:
        """Check if frames stopped arriving (window likely minimized)."""
        with self._lock:
            if self._frame_count == 0:
                return True
            return time.perf_counter() - self._last_frame_time > timeout

    def stop(self):
        """Stop the capture."""
        self._running = False
        if self._capture_control is not None:
            try:
                self._capture_control.stop()
            except Exception:
                pass
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count


class DxcamCapture:
    """
    Fallback: DXGI Desktop Duplication via bettercam.

    Faster than WGC but only captures visible screen content (full screen).
    Install: pip install bettercam
    """

    def __init__(self):
        self._camera = None
        self._running = False

    def start(self):
        try:
            import bettercam
            self._camera = bettercam.create(output_color="BGR")
            self._camera.start(target_fps=64)
            self._running = True
            print("[DxcamCapture] Started (bettercam DXGI)")
        except ImportError:
            print("[DxcamCapture] ERROR: bettercam not installed.")
            print("  Install: pip install bettercam")
            self._running = False
        except Exception as e:
            print(f"[DxcamCapture] ERROR: {e}")
            self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Grab latest frame. Returns BGR numpy array or None."""
        if self._camera is None:
            return None
        try:
            frame = self._camera.get_latest_frame()
            return frame  # Already BGR numpy array
        except Exception:
            return None

    def stop(self):
        self._running = False
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

    @property
    def is_running(self) -> bool:
        return self._running


def create_capture(backend: str = "wgc", window_name: str = "Counter-Strike 2"):
    """Factory function to create the appropriate capture backend."""
    if backend == "wgc":
        return WGCCapture(window_name=window_name)
    elif backend == "dxcam":
        return DxcamCapture()
    else:
        raise ValueError(f"Unknown capture backend: {backend}")
