"""
Console Log Parser — reads CS2 console.log (requires -condebug launch option).

Parses:
  - "CGameRules - paused on tick XXXXX" → current server tick
  - "server_start_tick: XXXXX" from demo_info → offset for demo ticks
"""

import os
import re
from typing import Optional


# Patterns
_RE_PAUSED_TICK = re.compile(r"CGameRules - paused on tick (\d+)")
_RE_SERVER_START_TICK = re.compile(r"server_start_tick:\s*(\d+)")


class ConsoleLogReader:
    """Reads and parses CS2 console.log file."""

    def __init__(self, log_path: str):
        self._path = log_path
        self._last_size: int = 0

    @property
    def path(self) -> str:
        return self._path

    @property
    def exists(self) -> bool:
        return os.path.isfile(self._path)

    def mark_position(self):
        """Mark current end of file so next read only gets new lines."""
        if self.exists:
            self._last_size = os.path.getsize(self._path)
        else:
            self._last_size = 0

    def read_new_lines(self) -> str:
        """Read only lines appended since last mark_position()."""
        if not self.exists:
            return ""
        try:
            size = os.path.getsize(self._path)
            if size <= self._last_size:
                return ""
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._last_size)
                data = f.read()
            self._last_size = size
            return data
        except (OSError, IOError):
            return ""

    def parse_paused_tick(self, text: Optional[str] = None) -> Optional[int]:
        """Parse the LAST 'paused on tick' from text (or new lines)."""
        if text is None:
            text = self.read_new_lines()
        matches = _RE_PAUSED_TICK.findall(text)
        if matches:
            return int(matches[-1])  # last occurrence
        return None

    def parse_server_start_tick(self, text: Optional[str] = None) -> Optional[int]:
        """Parse server_start_tick from demo_info output."""
        if text is None:
            text = self.read_new_lines()
        match = _RE_SERVER_START_TICK.search(text)
        if match:
            return int(match.group(1))
        return None

    def wait_for_paused_tick(self, timeout: float = 3.0) -> Optional[int]:
        """Read new lines repeatedly until paused tick appears or timeout."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            tick = self.parse_paused_tick()
            if tick is not None:
                return tick
            time.sleep(0.05)
        return None

    def wait_for_demo_playing(self, timeout: float = 30.0) -> bool:
        """
        Wait until CS2 writes 'Achievements disabled: demo playing.' to log.

        CS2 writes this line every time demo playback starts.
        Returns True if detected within timeout, False otherwise.
        """
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.read_new_lines()
            if "Achievements disabled: demo playing" in text:
                return True
            time.sleep(0.2)
        return False
