import win32gui
import win32api
import win32con
import ctypes
import time
import random


# ================= HOTKEY =================
HOTKEY_ID = 1
MOD_NONE = 0
VK_F8 = win32con.VK_F8

# ================= WINDOW =================
hwnd = win32gui.FindWindow(None, "Counter-Strike 2")
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
print(left, top, right, bottom)

target_x = random(0, right - left - 1)
target_y = random(0, bottom - top - 1)

screen_x = left + target_x
screen_y = top + target_y

# ================= SCREEN =================
screen_w = win32api.GetSystemMetrics(78)
screen_h = win32api.GetSystemMetrics(79)
screen_left = win32api.GetSystemMetrics(76)
screen_top = win32api.GetSystemMetrics(77)

abs_x = int((screen_x - screen_left) * 65535 / screen_w)
abs_y = int((screen_y - screen_top) * 65535 / screen_h)

# ================= SENDINPUT =================
SendInput = ctypes.windll.user32.SendInput

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]

def move_abs(x, y):
    inp = INPUT(
        INPUT_MOUSE,
        MOUSEINPUT(
            x, y, 0,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
            0, None
        )
    )
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def get_cursor_abs():
    x, y = win32api.GetCursorPos()
    abs_x = int((x - screen_left) * 65535 / screen_w)
    abs_y = int((y - screen_top) * 65535 / screen_h)
    return abs_x, abs_y

def smooth_move(target_abs_x, target_abs_y, steps=25, delay=0.002):
    start_x, start_y = get_cursor_abs()
    for i in range(steps):
        t = (i + 1) / steps
        x = int(start_x + (target_abs_x - start_x) * t)
        y = int(start_y + (target_abs_y - start_y) * t)
        move_abs(x, y)
        time.sleep(delay)

# ================= REGISTER HOTKEY =================
if not win32api.RegisterHotKey(None, HOTKEY_ID, MOD_NONE, VK_F8):
    raise RuntimeError("Не удалось зарегистрировать hotkey")

print("Нажми F8 для наведения")

# ================= MESSAGE LOOP =================
try:
    while True:
        msg = win32gui.GetMessage(None, 0, 0)
        if msg[1][1] == win32con.WM_HOTKEY:
            smooth_move(abs_x, abs_y)
except KeyboardInterrupt:
    print("error")
finally:
    win32api.UnregisterHotKey(None, HOTKEY_ID)
# a = 39
# d = 64
# ans = []
# while a > 0:
#     if a // d > 0:
#         ans.append(d)
#         a -= d
#     else:
#         d //= 2
# print(ans)