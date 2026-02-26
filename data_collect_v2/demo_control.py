"""
Demo Controller — automates CS2 demo playback via direct console commands.

Types commands directly into CS2 console (no cfg files, no key binds).
Uses win32gui for window management (ensure not minimized for WGC).

Tick synchronization:
  Requires CS2 launch option: -condebug -conclearlog
  Reads console.log to get real tick via "CGameRules - paused on tick XXXXX"
"""

import time
from typing import Optional

from cs2_cmd import CS2Commander
from console_log import ConsoleLogReader


def _find_cs2_window(window_name: str = "Counter-Strike 2",
                     sandbox_name: Optional[str] = None):
    """Find CS2 window HWND by title."""
    try:
        import win32gui

        if sandbox_name:
            target = f"[{sandbox_name}#] {window_name}"
        else:
            target = window_name

        def callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if target in title:
                    results.append(hwnd)

        results = []
        win32gui.EnumWindows(callback, results)
        return results[0] if results else None
    except ImportError:
        return None


class DemoController:
    """Controls CS2 demo playback through direct console commands."""

    def __init__(
        self,
        cs2_window_name: str = "Counter-Strike 2",
        demo_dir: str = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo",
        sandbox_name: Optional[str] = None,
    ):
        self._window_name = cs2_window_name
        self._demo_dir = demo_dir
        self._sandbox_name = sandbox_name

        self._commander = CS2Commander(cs2_window_name, sandbox_name)
        self._user_id: Optional[int] = None
        self._console_log: Optional[ConsoleLogReader] = None
        self._server_start_tick: int = 0

    # ---- Connection ----

    def connect(self) -> bool:
        """Find CS2 window."""
        return self._commander.connect()

    @property
    def is_connected(self) -> bool:
        return self._commander.is_connected

    # ---- Window management (WGC needs non-minimized window) ----

    def ensure_not_minimized(self) -> bool:
        """If CS2 is minimized, restore it. WGC can't capture minimized windows."""
        try:
            import win32gui
            import win32con

            hwnd = _find_cs2_window(self._window_name, self._sandbox_name)
            if hwnd is None:
                return False

            if win32gui.IsIconic(hwnd):
                print("[DemoCtrl] CS2 is minimized! Restoring for WGC...")
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.3)
                self.send_to_back()
            return True
        except Exception:
            return True

    def send_to_back(self):
        """Move CS2 window behind other windows."""
        try:
            import win32gui
            import win32con

            hwnd = _find_cs2_window(self._window_name, self._sandbox_name)
            if hwnd:
                win32gui.SetWindowPos(
                    hwnd, win32con.HWND_BOTTOM,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                )
        except Exception:
            pass

    # ---- Demo control ----

    def load_demo(self, demo_filename: str, user_id: int, timeout: float = 20.0):
        """Load a demo and set up spectator mode."""
        self._user_id = user_id

        # Strip .dem extension (Source 2 accepts both, but safer without)
        stem = demo_filename[:-4] if demo_filename.endswith('.dem') else demo_filename

        # Stop any current demo for a clean state
        self._commander.send("stopdemo")
        time.sleep(1.5)

        # Mark log position before playdemo so we can detect when it loads
        if self._console_log:
            self._console_log.mark_position()

        self._commander.send(f"playdemo {stem}")

        # Wait for "Achievements disabled: demo playing" in console.log
        if self._console_log:
            print(f"[DemoCtrl] Waiting for demo to load (timeout={timeout}s)...")
            if self._console_log.wait_for_demo_playing(timeout):
                print(f"[DemoCtrl] Demo loaded: {demo_filename}")
            else:
                print(f"[DemoCtrl] WARNING: Demo load not confirmed within {timeout}s. "
                      "Continuing anyway (maybe console.log is slow).")
        else:
            time.sleep(15)  # Fallback: blind wait

        self._commander.send_batch([
            "demoui",
            "spec_show_xray 0",
            f"spec_player {user_id}",
        ])
        time.sleep(0.5)
        print(f"[DemoCtrl] Spectating user_id={user_id}")

    def goto_tick_and_pause(self, tick: int):
        """
        Pause demo, seek to tick.

        CS2 writes "CGameRules - paused on tick X" to console.log after seek.
        Marks log position BEFORE sending commands so read_paused_tick() can pick it up.
        Caller must call resume() when ready to start capture.
        """
        if self._console_log:
            self._console_log.mark_position()
        cmds = ["demo_pause", f"demo_gototick {tick}"]
        if self._user_id is not None:
            cmds.append(f"spec_player {self._user_id}")
        self._commander.send_batch(cmds)
        time.sleep(3)  # Wait for seek to complete

    def setup_console_log(self, log_path: str):
        """Initialize console log reader for tick sync."""
        self._console_log = ConsoleLogReader(log_path)
        if not self._console_log.exists:
            print(f"[DemoCtrl] WARNING: {log_path} not found. "
                  "Add -condebug -conclearlog to CS2 launch options.")
            self._console_log = None

    def parse_server_start_tick(self):
        """Parse server_start_tick from demo_info output in console.log."""
        if not self._console_log:
            return
        self._console_log.mark_position()
        self._commander.send("demo_info")
        time.sleep(1)
        text = self._console_log.read_new_lines()
        tick = self._console_log.parse_server_start_tick(text)
        if tick is not None:
            self._server_start_tick = tick
            print(f"[DemoCtrl] server_start_tick = {tick}")
        else:
            print("[DemoCtrl] WARNING: Could not parse server_start_tick")

    def read_paused_tick(self, timeout: float = 3.0) -> Optional[int]:
        """
        Read the "paused on tick X" written by the most recent pause/gototick.

        Assumes mark_position() was already called (by goto_tick_and_pause or manually).
        Returns demo-relative tick (server_tick - server_start_tick), or None.
        """
        if not self._console_log:
            return None

        server_tick = self._console_log.wait_for_paused_tick(timeout)
        if server_tick is None:
            print("[DemoCtrl] WARNING: Could not read tick from console.log")
            return None

        return server_tick - self._server_start_tick

    def pause_and_read_tick(self, timeout: float = 3.0) -> Optional[int]:
        """
        For mid-playback tick measurement: mark log → pause → read tick.

        Used at round end for drift measurement.
        Does NOT resume — caller must call resume().
        """
        if not self._console_log:
            return None

        self._console_log.mark_position()
        self._commander.send("demo_pause")
        return self.read_paused_tick(timeout)

    def resume(self):
        """Resume demo playback."""
        self._commander.send("demo_resume")

    def pause(self):
        """Pause demo playback."""
        self._commander.send("demo_pause")

    def stop_demo(self):
        """Stop current demo."""
        self._commander.send("stopdemo")

    def cleanup(self):
        """Close commander."""
        self._commander.close()
