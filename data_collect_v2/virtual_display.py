"""
Virtual Display — move CS2 to a virtual monitor so it doesn't interfere with the user.

Requires: Virtual Display Driver (IddSampleDriver) installed.
  https://github.com/itsmikethetech/Virtual-Display-Driver

The virtual display appears as a regular monitor in Windows.
WGC captures it normally since the window is rendered on a valid display surface.

Usage:
    vd = VirtualDisplay()
    monitors = vd.list_monitors()
    vd.move_window_to_monitor(monitor_index=1)  # Move CS2 to second monitor
"""

import ctypes
import ctypes.wintypes
from typing import List, Dict, Optional, Tuple


def list_monitors() -> List[Dict]:
    """
    Enumerate all connected monitors (real + virtual).

    Returns list of dicts:
        [
            {
                'index': 0,
                'name': r'\\.\DISPLAY1',
                'x': 0, 'y': 0, 'w': 1920, 'h': 1080,
                'primary': True,
                'device': 'NVIDIA GeForce RTX 3050'
            },
            ...
        ]
    """
    try:
        import win32api
        import win32con

        monitors = []
        for i, mon in enumerate(win32api.EnumDisplayMonitors(None, None)):
            hmon = mon[0]
            info = win32api.GetMonitorInfo(hmon)
            rect = info['Monitor']  # (left, top, right, bottom)
            work = info['Work']
            is_primary = bool(info['Flags'] & 1)  # MONITORINFOF_PRIMARY

            # Get device name
            device_name = info.get('Device', '')

            # Get adapter info
            adapter = ''
            if device_name:
                try:
                    dev = win32api.EnumDisplayDevices(device_name, 0, 0)
                    adapter = dev.DeviceString
                except Exception:
                    pass

            monitors.append({
                'index': i,
                'name': device_name,
                'x': rect[0],
                'y': rect[1],
                'w': rect[2] - rect[0],
                'h': rect[3] - rect[1],
                'primary': is_primary,
                'device': adapter,
            })

        return monitors

    except ImportError:
        print("[VirtualDisplay] ERROR: pywin32 not installed")
        return []


def find_virtual_monitor(monitors: List[Dict]) -> Optional[int]:
    """
    Auto-detect a virtual display monitor.

    Heuristics (in order):
    1. Monitor with 'IddSampleDriver' or 'Virtual' in device name
    2. Non-primary monitor (if exactly one exists)

    Returns monitor index or None.
    """
    # Heuristic 1: Look for virtual display driver keywords
    virtual_keywords = ['virtual', 'idd', 'iddsample', 'headless', 'dummy']
    for mon in monitors:
        device_lower = mon['device'].lower()
        name_lower = mon['name'].lower()
        combined = device_lower + ' ' + name_lower
        if any(kw in combined for kw in virtual_keywords):
            return mon['index']

    # Heuristic 2: If exactly one non-primary monitor, assume it's the virtual one
    non_primary = [m for m in monitors if not m['primary']]
    if len(non_primary) == 1:
        return non_primary[0]['index']

    return None


def print_monitors(monitors: List[Dict]):
    """Print monitor list in a readable format."""
    print(f"\nDetected {len(monitors)} monitor(s):")
    print("-" * 70)
    for m in monitors:
        flags = []
        if m['primary']:
            flags.append('PRIMARY')
        device_lower = (m['device'] + ' ' + m['name']).lower()
        if any(kw in device_lower for kw in ['virtual', 'idd', 'dummy', 'headless']):
            flags.append('VIRTUAL')
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"  [{m['index']}] {m['name']}: {m['w']}x{m['h']} at ({m['x']},{m['y']})"
              f"  — {m['device'] or 'Unknown'}{flag_str}")
    print("-" * 70)


def move_window_to_monitor(
    window_name: str,
    monitor_index: int,
    monitors: Optional[List[Dict]] = None,
    borderless: bool = True,
) -> bool:
    """
    Move a window to the specified monitor.

    Args:
        window_name: Window title substring (e.g. "Counter-Strike 2")
        monitor_index: Index from list_monitors()
        monitors: Pre-fetched monitor list (or None to re-enumerate)
        borderless: If True, remove window borders and resize to fill monitor

    Returns True on success.
    """
    try:
        import win32gui
        import win32con

        if monitors is None:
            monitors = list_monitors()

        if monitor_index < 0 or monitor_index >= len(monitors):
            print(f"[VirtualDisplay] Invalid monitor index {monitor_index} "
                  f"(have {len(monitors)} monitors)")
            return False

        mon = monitors[monitor_index]

        # Find window
        hwnd = _find_window(window_name)
        if hwnd is None:
            print(f"[VirtualDisplay] Window '{window_name}' not found")
            return False

        # Restore if minimized
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            import time
            time.sleep(0.3)

        if borderless:
            # Remove window borders for borderless fullscreen on the virtual monitor
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            # Remove title bar and borders
            style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME |
                        win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX |
                        win32con.WS_SYSMENU)
            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

            # Remove extended style borders
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            ex_style &= ~(win32con.WS_EX_DLGMODALFRAME | win32con.WS_EX_CLIENTEDGE |
                          win32con.WS_EX_STATICEDGE)
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

            # Move and resize to fill the monitor
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_TOP,
                mon['x'], mon['y'], mon['w'], mon['h'],
                win32con.SWP_FRAMECHANGED | win32con.SWP_NOACTIVATE
            )
        else:
            # Just move the window to the monitor's top-left
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_TOP,
                mon['x'], mon['y'], 0, 0,
                win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
            )

        print(f"[VirtualDisplay] Moved '{window_name}' to monitor [{monitor_index}] "
              f"({mon['w']}x{mon['h']} at {mon['x']},{mon['y']})")
        return True

    except ImportError:
        print("[VirtualDisplay] ERROR: pywin32 not installed")
        return False
    except Exception as e:
        print(f"[VirtualDisplay] ERROR: {e}")
        return False


def restore_window_style(window_name: str) -> bool:
    """Restore window borders (undo borderless mode)."""
    try:
        import win32gui
        import win32con

        hwnd = _find_window(window_name)
        if hwnd is None:
            return False

        # Restore standard overlapped window style
        style = (win32con.WS_OVERLAPPED | win32con.WS_CAPTION |
                 win32con.WS_SYSMENU | win32con.WS_THICKFRAME |
                 win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX |
                 win32con.WS_VISIBLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

        win32gui.SetWindowPos(
            hwnd, 0, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
            win32con.SWP_FRAMECHANGED | win32con.SWP_NOZORDER
        )
        return True
    except Exception:
        return False


def _find_window(title_substring: str) -> Optional[int]:
    """Find window HWND by title substring."""
    try:
        import win32gui

        result = [None]

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title_substring in title:
                    result[0] = hwnd

        win32gui.EnumWindows(callback, None)
        return result[0]
    except Exception:
        return None


def lock_cursor_to_physical(virtual_monitor_index: Optional[int] = None) -> bool:
    """
    Lock mouse cursor to all physical monitors (exclude the virtual one).

    Uses ClipCursor with a bounding rect that covers all monitors except
    the virtual display. Works correctly with 1, 2, or more physical monitors.

    Call unlock_cursor() to release.
    """
    try:
        monitors = list_monitors()
        if not monitors:
            return False

        # Determine which monitor is virtual
        if virtual_monitor_index is None:
            virtual_monitor_index = find_virtual_monitor(monitors)

        # Get all physical monitors (everything except the virtual one)
        physical = [m for m in monitors if m['index'] != virtual_monitor_index]
        if not physical:
            return False

        # Compute bounding rect of all physical monitors
        left = min(m['x'] for m in physical)
        top = min(m['y'] for m in physical)
        right = max(m['x'] + m['w'] for m in physical)
        bottom = max(m['y'] + m['h'] for m in physical)

        rect = ctypes.wintypes.RECT(left, top, right, bottom)
        ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

        names = ', '.join(m['name'] for m in physical)
        print(f"[VirtualDisplay] Cursor locked to physical monitors: {names} "
              f"({right - left}x{bottom - top})")
        return True
    except Exception as e:
        print(f"[VirtualDisplay] Failed to lock cursor: {e}")
        return False


def unlock_cursor():
    """Release cursor lock (allow movement to all monitors)."""
    try:
        ctypes.windll.user32.ClipCursor(None)
        print("[VirtualDisplay] Cursor unlocked")
    except Exception:
        pass


def setup_virtual_display(
    window_name: str = "Counter-Strike 2",
    monitor_index: Optional[int] = None,
) -> bool:
    """
    High-level function: detect virtual monitor and move CS2 to it.

    Args:
        window_name: CS2 window title
        monitor_index: Specific monitor index, or None for auto-detect

    Returns True on success.
    """
    monitors = list_monitors()
    if not monitors:
        print("[VirtualDisplay] No monitors detected")
        return False

    print_monitors(monitors)

    if monitor_index is None:
        monitor_index = find_virtual_monitor(monitors)
        if monitor_index is None:
            print("[VirtualDisplay] Could not auto-detect virtual monitor.")
            print("  Install Virtual Display Driver: https://github.com/itsmikethetech/Virtual-Display-Driver")
            print("  Or specify --virtual-display <index> with a monitor index from the list above.")
            return False
        print(f"[VirtualDisplay] Auto-detected virtual monitor: [{monitor_index}]")
    else:
        if monitor_index >= len(monitors):
            print(f"[VirtualDisplay] Monitor index {monitor_index} out of range (0-{len(monitors)-1})")
            return False

    return move_window_to_monitor(window_name, monitor_index, monitors, borderless=True)
