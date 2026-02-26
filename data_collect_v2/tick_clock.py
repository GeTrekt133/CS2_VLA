"""
Tick Clock — maps wall-clock time to CS2 game ticks during demo playback.

At tickrate=64 and playback_speed=1.0, each tick = 15.625ms.
perf_counter precision ~1ms → error < 1 tick.

Supports calibration via real tick from console.log (-condebug).
"""

import time
from typing import Optional


class TickClock:

    def __init__(
        self,
        tickrate: int = 64,
        playback_speed: float = 1.0,
        tick_stride: int = 1,
    ):
        self.tickrate = tickrate
        self.playback_speed = playback_speed
        self.tick_stride = tick_stride

        self._wall_start: float = 0.0
        self._tick_start: int = 0
        self._running: bool = False

        # Drift tracking
        self._calibrated_start: Optional[int] = None  # real tick at round start
        self._calibrated_end: Optional[int] = None     # real tick at round end
        self._expected_end: Optional[int] = None        # expected tick at round end
        self._drift: int = 0                            # end_expected - end_real

    def start_round(self, start_tick: int):
        """Begin tracking from start_tick at current wall-clock time."""
        self._tick_start = start_tick
        self._wall_start = time.perf_counter()
        self._running = True
        self._calibrated_start = start_tick
        self._calibrated_end = None
        self._expected_end = None
        self._drift = 0

    def stop(self):
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def drift(self) -> int:
        """Drift in ticks (positive = clock ran ahead of demo)."""
        return self._drift

    @property
    def drift_info(self) -> dict:
        """Return drift metadata for dataset.json."""
        return {
            "start_tick_real": self._calibrated_start,
            "end_tick_real": self._calibrated_end,
            "end_tick_expected": self._expected_end,
            "drift_ticks": self._drift,
        }

    def record_end_tick(self, real_tick: int):
        """Record real tick at round end for drift calculation."""
        self._calibrated_end = real_tick
        self._expected_end = self.current_tick()
        self._drift = self._expected_end - real_tick
        drift_ms = self._drift * 1000 / self.tickrate
        print(f"[TickClock] Drift: {self._drift} ticks ({drift_ms:.0f} ms)")

    def current_tick(self) -> int:
        """Estimate current game tick from elapsed wall-clock time."""
        elapsed = time.perf_counter() - self._wall_start
        raw = self._tick_start + int(elapsed * self.tickrate * self.playback_speed)
        return raw

    def should_capture(self, last_saved_tick: int) -> Optional[int]:
        """
        Return the NEXT tick to capture (last_saved + stride) if the clock
        has already reached it, otherwise None.

        Returns last_saved+stride (not current tick) so that even if the poll
        misses a boundary and current is 2+ ticks ahead, we step one tick at
        a time and the loop catches up without skipping any ticks.
        """
        next_tick = last_saved_tick + self.tick_stride
        if self.current_tick() >= next_tick:
            return next_tick
        return None

    def seconds_until(self, target_tick: int) -> float:
        """Wall-clock seconds until target_tick is reached."""
        current = self.current_tick()
        delta = target_tick - current
        if delta <= 0:
            return 0.0
        return delta / (self.tickrate * self.playback_speed)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._wall_start
