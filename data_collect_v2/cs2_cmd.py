"""
CS2 Commander — sends console commands by typing directly into CS2 console.

How it works:
  1. Focus CS2 window (with Alt trick for reliable foreground)
  2. Open console (backtick key `)
  3. Type command via hardware scan codes (layout-independent)
  4. Press Enter
  5. Close console (backtick key `)

Supports Sandboxie-Plus: window title becomes "[SandboxName#] Counter-Strike 2".
No cfg files, no F10 bind, no autoexec modification needed.
"""

import time
import ctypes
import ctypes.wintypes
import threading
from typing import Optional, List

# ---- Win32 constants ----
VK_MENU = 0x12        # Alt key
VK_SHIFT = 0x10
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ---- US QWERTY scan codes (physical key positions — layout independent) ----
_SCAN = {
    'a': 0x1E, 'b': 0x30, 'c': 0x2E, 'd': 0x20, 'e': 0x12, 'f': 0x21,
    'g': 0x22, 'h': 0x23, 'i': 0x17, 'j': 0x24, 'k': 0x25, 'l': 0x26,
    'm': 0x32, 'n': 0x31, 'o': 0x18, 'p': 0x19, 'q': 0x10, 'r': 0x13,
    's': 0x1F, 't': 0x14, 'u': 0x16, 'v': 0x2F, 'w': 0x11, 'x': 0x2D,
    'y': 0x15, 'z': 0x2C,
    '0': 0x0B, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06,
    '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A,
    ' ': 0x39, '-': 0x0C, '=': 0x0D, '[': 0x1A, ']': 0x1B,
    ';': 0x27, "'": 0x28, '`': 0x29, '\\': 0x2B, ',': 0x33,
    '.': 0x34, '/': 0x35,
}

# Characters that require Shift + base key on US QWERTY
_SHIFT_CHARS = {
    '"': "'", '_': '-', '+': '=', '{': '[', '}': ']',
    ':': ';', '~': '`', '|': '\\', '<': ',', '>': '.', '?': '/',
    '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
    '^': '6', '&': '7', '*': '8', '(': '9', ')': '0',
}

# Key scan codes
CONSOLE_SCAN = 0x29  # ` (backtick/tilde) — console toggle
ENTER_SCAN = 0x1C


def _find_window(window_name: str, sandbox_name: Optional[str] = None) -> Optional[int]:
    """Find window HWND by title.

    If sandbox_name is set, looks for Sandboxie prefix: [SandboxName#] Window Title
    """
    if sandbox_name:
        target = f"[{sandbox_name}#] {window_name}"
    else:
        target = window_name

    result = []

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if target in title:
                    result.append(hwnd)
        return True

    user32.EnumWindows(enum_callback, 0)
    return result[0] if result else None


class CS2Commander:
    """
    Sends console commands to CS2 by typing directly into the console.

    Focus window -> open console -> type -> Enter -> close console.
    Works with Sandboxie-Plus for isolated/parallel instances.
    """

    def __init__(self, window_name: str = "Counter-Strike 2",
                 sandbox_name: Optional[str] = None):
        self._window_name = window_name
        self._sandbox_name = sandbox_name
        self._hwnd: Optional[int] = None
        self._lock = threading.Lock()

    def connect(self, retries: int = 3, delay: float = 2.0) -> bool:
        """Find CS2 window. No bind setup needed."""
        for attempt in range(retries):
            self._hwnd = _find_window(self._window_name, self._sandbox_name)
            if self._hwnd:
                prefix = f"[{self._sandbox_name}#] " if self._sandbox_name else ""
                print(f"[CS2Cmd] Found: {prefix}{self._window_name} (HWND={self._hwnd:#x})")
                return True

            print(f"[CS2Cmd] Window not found (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(delay)

        target = f"[{self._sandbox_name}#] {self._window_name}" if self._sandbox_name else self._window_name
        print(f"[CS2Cmd] FAILED: '{target}' not found. Is CS2 running?")
        return False

    # ---- Send commands ----

    def send(self, command: str) -> bool:
        """Type a single command into CS2 console."""
        return self.send_batch([command])

    def send_batch(self, commands: List[str]) -> bool:
        """Open console, type all commands, close console."""
        with self._lock:
            if not self._hwnd:
                return False

            # Check window still exists
            if not user32.IsWindow(self._hwnd):
                print("[CS2Cmd] Window lost, reconnecting...")
                self._hwnd = _find_window(self._window_name, self._sandbox_name)
                if not self._hwnd:
                    print("[CS2Cmd] CS2 window not found")
                    return False

            # Focus CS2
            self._set_foreground(self._hwnd)
            time.sleep(0.05)

            # Open console
            self._press_scan(CONSOLE_SCAN)
            time.sleep(0.1)

            # Type each command + Enter
            for cmd in commands:
                self._type_scancode(cmd)
                time.sleep(0.02)
                self._press_scan(ENTER_SCAN)
                time.sleep(0.05)

            # Close console
            self._press_scan(CONSOLE_SCAN)
            time.sleep(0.05)

            return True

    # ---- Input helpers ----

    def _set_foreground(self, hwnd):
        """Reliably bring window to foreground (bypasses Windows restrictions)."""
        fg = user32.GetForegroundWindow()
        if fg == hwnd:
            return

        fg_thread = user32.GetWindowThreadProcessId(fg, None)
        our_thread = kernel32.GetCurrentThreadId()

        attached = False
        if fg_thread != our_thread:
            user32.AttachThreadInput(our_thread, fg_thread, True)
            attached = True

        # Alt key trick — allows SetForegroundWindow from background process
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)

        if attached:
            user32.AttachThreadInput(our_thread, fg_thread, False)

    def _press_scan(self, scan_code: int):
        """Press and release a key by scan code."""
        user32.keybd_event(0, scan_code, KEYEVENTF_SCANCODE, 0)
        time.sleep(0.02)
        user32.keybd_event(0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0)
        time.sleep(0.02)

    def _type_scancode(self, text: str):
        """Type ASCII text via hardware scan codes.

        Source 2 console reads scan codes as US QWERTY positions,
        so this works regardless of the active OS keyboard layout.
        """
        for char in text:
            shift = False

            if char.isupper():
                scan = _SCAN.get(char.lower())
                shift = True
            elif char in _SHIFT_CHARS:
                scan = _SCAN.get(_SHIFT_CHARS[char])
                shift = True
            else:
                scan = _SCAN.get(char)

            if scan is None:
                continue

            if shift:
                user32.keybd_event(VK_SHIFT, 0x2A, 0, 0)
                time.sleep(0.01)

            user32.keybd_event(0, scan, KEYEVENTF_SCANCODE, 0)
            time.sleep(0.02)
            user32.keybd_event(0, scan, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0)

            if shift:
                time.sleep(0.01)
                user32.keybd_event(VK_SHIFT, 0x2A, KEYEVENTF_KEYUP, 0)

            time.sleep(0.01)

    # ---- Lifecycle ----

    def close(self):
        self._hwnd = None

    @property
    def is_connected(self) -> bool:
        return self._hwnd is not None and bool(user32.IsWindow(self._hwnd))

    @property
    def hwnd(self) -> Optional[int]:
        return self._hwnd
