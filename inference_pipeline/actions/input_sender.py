"""
Low-level Windows SendInput wrapper using ctypes.
Provides mouse movement and keyboard input for games.
"""

import ctypes
from ctypes import wintypes
from typing import Optional

# Windows API constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Mouse event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000

# X button identifiers
XBUTTON1 = 0x0001  # mouse4
XBUTTON2 = 0x0002  # mouse5

# Keyboard event flags
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001


# === ctypes structures for SendInput ===

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort)
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUT_UNION)
    ]


class InputSender:
    """
    Low-level Windows SendInput wrapper for mouse and keyboard.

    Uses scan codes for keyboard input (more reliable for games).
    Supports mouse movement and buttons including side buttons.
    """

    # Scan codes for common keys (hardware-level, more reliable for games)
    SCAN_CODES = {
        # Movement
        'w': 0x11,
        's': 0x1F,
        'a': 0x1E,
        'd': 0x20,
        'space': 0x39,
        'ctrl': 0x1D,
        'shift': 0x2A,
        'alt': 0x38,

        # Actions
        'r': 0x13,
        'e': 0x12,
        'f': 0x21,
        'g': 0x22,
        'q': 0x10,
        'c': 0x2E,
        'x': 0x2D,
        'z': 0x2C,
        'b': 0x30,
        'n': 0x31,
        'm': 0x32,
        'tab': 0x0F,
        'escape': 0x01,
        'enter': 0x1C,

        # Numbers
        '1': 0x02,
        '2': 0x03,
        '3': 0x04,
        '4': 0x05,
        '5': 0x06,
        '6': 0x07,
        '7': 0x08,
        '8': 0x09,
        '9': 0x0A,
        '0': 0x0B,

        # Console
        '`': 0x29,
        'tilde': 0x29,

        # Function keys
        'f1': 0x3B,
        'f2': 0x3C,
        'f3': 0x3D,
        'f4': 0x3E,
        'f5': 0x3F,
    }

    # Virtual key codes (fallback)
    VK_CODES = {
        'w': 0x57, 's': 0x53, 'a': 0x41, 'd': 0x44,
        'space': 0x20, 'ctrl': 0x11, 'shift': 0x10, 'alt': 0x12,
        'r': 0x52, 'e': 0x45, 'f': 0x46, 'g': 0x47,
        'q': 0x51, 'c': 0x43, 'x': 0x58, 'z': 0x5A,
        'b': 0x42, 'n': 0x4E, 'm': 0x4D,
        'tab': 0x09, 'escape': 0x1B, 'enter': 0x0D,
        '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35,
        '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39, '0': 0x30,
    }

    def __init__(self):
        """Initialize input sender."""
        self.user32 = ctypes.windll.user32
        self._extra = ctypes.pointer(ctypes.c_ulong(0))

    def send_mouse_move(self, dx: int, dy: int):
        """
        Send relative mouse movement.

        Args:
            dx: Horizontal movement (positive = right)
            dy: Vertical movement (positive = down)
        """
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = dx
        inp.union.mi.dy = dy
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE
        inp.union.mi.dwExtraInfo = self._extra

        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def send_mouse_button(self, button: str, down: bool):
        """
        Send mouse button press/release.

        Args:
            button: 'mouse1' (left), 'mouse2' (right), 'mouse3' (middle),
                   'mouse4' (side back), 'mouse5' (side forward)
            down: True for press, False for release
        """
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dwExtraInfo = self._extra

        if button == 'mouse1':
            inp.union.mi.dwFlags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP

        elif button == 'mouse2':
            inp.union.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP

        elif button == 'mouse3':
            inp.union.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN if down else MOUSEEVENTF_MIDDLEUP

        elif button == 'mouse4':
            inp.union.mi.dwFlags = MOUSEEVENTF_XDOWN if down else MOUSEEVENTF_XUP
            inp.union.mi.mouseData = XBUTTON1

        elif button == 'mouse5':
            inp.union.mi.dwFlags = MOUSEEVENTF_XDOWN if down else MOUSEEVENTF_XUP
            inp.union.mi.mouseData = XBUTTON2

        else:
            return  # Unknown button

        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def send_key(self, key: str, down: bool):
        """
        Send keyboard key press/release.

        Args:
            key: Key name (e.g., 'w', 'space', 'ctrl', '1', 'mouse1')
            down: True for press, False for release
        """
        # Handle mouse buttons
        if key.startswith('mouse'):
            return self.send_mouse_button(key, down)

        # Get scan code
        scan = self.SCAN_CODES.get(key.lower())
        if scan is None:
            # Try virtual key code as fallback
            vk = self.VK_CODES.get(key.lower())
            if vk is not None:
                self._send_key_vk(vk, down)
            return

        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wScan = scan
        inp.union.ki.dwFlags = KEYEVENTF_SCANCODE
        inp.union.ki.dwExtraInfo = self._extra

        if not down:
            inp.union.ki.dwFlags |= KEYEVENTF_KEYUP

        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _send_key_vk(self, vk: int, down: bool):
        """Send key using virtual key code (fallback)."""
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.dwFlags = 0 if down else KEYEVENTF_KEYUP
        inp.union.ki.dwExtraInfo = self._extra

        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def send_key_tap(self, key: str, duration: float = 0.05):
        """
        Send a quick key tap (press + release).

        Args:
            key: Key name
            duration: Time between press and release (seconds)
        """
        import time
        self.send_key(key, down=True)
        time.sleep(duration)
        self.send_key(key, down=False)

    def send_mouse_click(self, button: str = 'mouse1', duration: float = 0.05):
        """
        Send a quick mouse click.

        Args:
            button: Mouse button name
            duration: Time between press and release
        """
        import time
        self.send_mouse_button(button, down=True)
        time.sleep(duration)
        self.send_mouse_button(button, down=False)

    def type_string(self, text: str, delay: float = 0.02):
        """
        Type a string character by character.

        Args:
            text: String to type
            delay: Delay between characters
        """
        import time
        for char in text:
            key = char.lower()
            if key in self.SCAN_CODES:
                self.send_key_tap(key)
                time.sleep(delay)


def test_input_sender():
    """Test InputSender (be careful, this sends real inputs!)."""
    import time

    print("Testing InputSender...")
    print("WARNING: This will send real mouse/keyboard inputs!")
    print("Press Ctrl+C within 3 seconds to cancel...")

    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("Cancelled")
        return

    sender = InputSender()

    # Test mouse movement
    print("\nTesting mouse movement (small square)...")
    for _ in range(4):
        sender.send_mouse_move(50, 0)
        time.sleep(0.2)
        sender.send_mouse_move(0, 50)
        time.sleep(0.2)
        sender.send_mouse_move(-50, 0)
        time.sleep(0.2)
        sender.send_mouse_move(0, -50)
        time.sleep(0.2)

    print("Mouse test complete")

    # Don't test keyboard without explicit confirmation
    print("\nSkipping keyboard test for safety")
    print("To test keyboard, uncomment the code below")

    # # Test keyboard (uncomment to test)
    # print("\nTesting keyboard (typing 'test')...")
    # time.sleep(1)
    # sender.type_string("test")

    print("\nAll tests complete!")


if __name__ == "__main__":
    test_input_sender()
