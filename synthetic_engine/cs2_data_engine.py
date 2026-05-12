"""
CS2DataEngine — orchestrates synthetic data collection in CS2.

Workflow per scenario:
  1. Teleport player to position (noclip on, setpos, setang, noclip off)
  2. Teleport bot(s) to target positions
  3. Slow down game time (host_timescale 0.05)
  4. For each tick in trajectory:
     a. setang to trajectory angle
     b. Capture screenshot via WGC
     c. Record audio chunk
  5. Save frames + metadata

Requires CS2 running with sv_cheats 1 on a local server.
Uses cs2_cmd.py for console commands and capture.py for screenshots.
"""

import os
import sys
import time
import json
import math
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

# Add data_collect_v2 to path for reuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))

from cs2_cmd import CS2Commander
from capture import WGCCapture, create_capture
from frame_saver import FrameSaver

from scenario_generator import Scenario, ScenarioGenerator
from trajectory_generator import TrajectoryGenerator, TrajectoryPoint


class CS2DataEngine:
    """Orchestrates synthetic data collection inside CS2."""

    # Phase 1: initial cvars (work even during warmup/freeze)
    SETUP_CVARS = [
        "sv_cheats 1",
        "mp_autoteambalance 0",
        "mp_limitteams 0",
        "mp_freezetime 0",
        "mp_warmup_end",
        "mp_roundtime_defuse 999",
        "mp_roundtime 999",
        "mp_buy_anywhere 1",
        "sv_infinite_ammo 1",
        "bot_stop 1",
        "bot_dont_shoot 1",
        "cl_drawhud 1",
        "r_drawviewmodel 1",
    ]

    # Phase 2: restart the game so mp_freezetime 0 takes effect
    RESTART_COMMANDS = [
        "mp_restartgame 1",
    ]

    # Phase 3: after restart, apply per-player cheats
    PLAYER_COMMANDS = [
        "god",
        "noclip 0",
        "bot_stop 1",
        "bot_dont_shoot 1",
    ]

    def __init__(
        self,
        output_dir: str,
        cs2_window_name: str = "Counter-Strike 2",
        sandbox_name: Optional[str] = None,
        frame_size: int = 640,
        jpeg_quality: int = 85,
        capture_backend: str = "wgc",
        record_audio: bool = True,
        audio_sample_rate: int = 16000,
        timescale: float = 0.05,       # game slowdown factor
        tick_delay: float = 0.15,      # delay between setang commands (seconds)
    ):
        self.output_dir = output_dir
        self.frame_size = frame_size
        self.timescale = timescale
        self.tick_delay = tick_delay

        # Components
        self.cmd = CS2Commander(cs2_window_name, sandbox_name)
        self.capture = create_capture(capture_backend, cs2_window_name)
        self.saver = FrameSaver(
            num_threads=4,
            jpeg_quality=jpeg_quality,
            frame_size=frame_size,
        )
        # Audio disabled: host_timescale slows game sounds, making them unusable
        self.audio_recorder = None

        self._connected = False
        self._map_ready = False
        self._scenario_count = 0
        self._bot_userid = None  # set by _spawn_bots, used by setpos_player

    # ------------------------------------------------------------------ #
    # Connection & Setup                                                   #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """Connect to CS2 window."""
        ok = self.cmd.connect()
        if ok:
            self._connected = True
        return ok

    def setup_map(self, map_name: str = "de_dust2"):
        """Load map and configure server for data collection.

        Handles: existing bots, freeze time, warmup.
        CS2 must be running.
        """
        if not self._connected:
            raise RuntimeError("Not connected to CS2. Call connect() first.")

        print(f"[Engine] Loading map {map_name}...")
        self.cmd.send(f"map {map_name}")
        time.sleep(15)  # wait for map load + warmup

        # Phase 1: set cvars
        print("[Engine] Setting cvars...")
        self.cmd.send_batch(self.SETUP_CVARS)
        time.sleep(2)

        # Kick all existing bots — we'll add our own
        print("[Engine] Kicking existing bots...")
        self.cmd.send("bot_kick")
        time.sleep(2)

        # Phase 2: restart game so freezetime=0 takes effect
        print("[Engine] Restarting game (mp_restartgame 1)...")
        self.cmd.send_batch(self.RESTART_COMMANDS)
        time.sleep(3)  # wait for restart

        # Phase 3: apply per-player commands (god mode etc.)
        print("[Engine] Applying player commands...")
        self.cmd.send_batch(self.PLAYER_COMMANDS)
        time.sleep(1)

        self._map_ready = True
        print(f"[Engine] Map {map_name} ready.")

    def _teleport_player(self, x: float, y: float, z: float,
                         pitch: float, yaw: float):
        """Teleport player safely using noclip to avoid falling through map."""
        self.cmd.send_batch([
            "noclip 1",
            f"setpos {x:.1f} {y:.1f} {z + 10:.1f}",  # +10 units above ground
            f"setang {pitch:.2f} {yaw:.2f} 0",
        ])
        time.sleep(0.3)
        # Disable noclip — player drops onto ground naturally
        self.cmd.send("noclip 0")
        time.sleep(0.3)

    def _set_view_angle(self, pitch: float, yaw: float):
        """Set player view angle only (no position change)."""
        self.cmd.send(f"setang {pitch:.2f} {yaw:.2f} 0")

    def _spawn_bots(self, n: int = 1, bot_userid: int = 11):
        """Add enemy bots. Bots are frozen (bot_stop 1) and won't shoot.

        Args:
            bot_userid: userid for setpos_player (status shows id, CS2 needs id+1).
                        Default 11 = first bot (status id 10, +1).
        """
        for _ in range(n):
            self.cmd.send("bot_add_ct")
            time.sleep(1.5)
        # Re-apply freeze after adding
        self.cmd.send_batch(["bot_stop 1", "bot_dont_shoot 1"])
        time.sleep(0.5)
        self._bot_userid = bot_userid

    def _place_bot_at(self, x: float, y: float, z: float,
                      ducking: bool = False, **_kwargs):
        """Teleport bot to exact coordinates.

        Args:
            x, y, z: world position (feet)
            ducking: if True, lower Z by ~18 units to simulate crouch
        """
        if self._bot_userid is None:
            self.cmd.send("bot_place")
            time.sleep(0.5)
            return

        if ducking:
            z -= 18.0

        self.cmd.send(f"setpos_player {self._bot_userid} {x:.1f} {y:.1f} {z:.1f}")
        time.sleep(0.2)

    # ------------------------------------------------------------------ #
    # Capture helpers                                                      #
    # ------------------------------------------------------------------ #

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from WGC capture."""
        frame = self.capture.get_frame()
        if frame is None:
            return None
        return frame

    def _wait_for_new_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Wait for a fresh frame after setang command."""
        time.sleep(self.tick_delay)
        return self._capture_frame()

    # ------------------------------------------------------------------ #
    # Run scenario                                                         #
    # ------------------------------------------------------------------ #

    def run_scenario(
        self,
        scenario: Scenario,
        trajectory: List[TrajectoryPoint],
        scenario_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single scenario and save frames + metadata.

        Returns metadata dict for DataExporter, or None on failure.
        """
        if not self._map_ready:
            raise RuntimeError("Map not ready. Call setup_map() first.")

        if scenario_id is None:
            scenario_id = f"synth_{self._scenario_count:06d}"
        self._scenario_count += 1

        # Create output directory for this scenario
        scenario_dir = os.path.join(self.output_dir, "frames", scenario_id)
        audio_dir = os.path.join(self.output_dir, "audio", scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        if self.audio_recorder:
            os.makedirs(audio_dir, exist_ok=True)

        print(f"[Engine] Running {scenario.scenario_type} scenario {scenario_id} "
              f"({len(trajectory)} ticks, delta_yaw={scenario.delta_yaw:.1f})")

        # 1. Normal timescale for teleportation (faster)
        self.cmd.send("host_timescale 1")
        time.sleep(0.3)

        # 2. Teleport player to start position
        self._teleport_player(
            *scenario.player_pos,
            scenario.initial_pitch,
            scenario.initial_yaw,
        )
        time.sleep(0.5)

        # 3. Teleport bot to target position with rotation/duck
        for i, bot in enumerate(scenario.bots):
            if bot.label.startswith("tracking_end"):
                continue
            self._place_bot_at(
                *bot.pos,
                yaw=getattr(bot, 'victim_yaw', 0.0),
                pitch=getattr(bot, 'victim_pitch', 0.0),
                ducking=getattr(bot, 'ducking', False),
            )

        # 5. Slow down time for frame capture
        self.cmd.send(f"host_timescale {self.timescale}")
        time.sleep(0.5)

        # 6. Start audio recording
        audio_path = None
        if self.audio_recorder:
            audio_path = os.path.join(audio_dir, "round_1.wav")
            self.audio_recorder.start_recording(audio_path)

        # 7. Capture frames along trajectory
        states = []
        tick_base = 1000  # synthetic tick numbering

        for i, point in enumerate(trajectory):
            tick = tick_base + i

            # Set view angle
            self._set_view_angle(point.pitch, point.yaw)

            # Wait and capture
            frame = self._wait_for_new_frame()
            if frame is not None:
                self.saver.save(frame, tick, scenario_dir)

            # Build state dict (same format as DatasetIntent expects)
            state = {
                "tick": tick,
                "keys": list(point.keys),
                "mouse": [point.yaw, point.pitch],
                "detects": [],
                "hp": 100,
                "armor": 100,
                "helmet": True,
                "defuser": False,
                "weapon": "AK-47",
                "ammo": 30,
                "weapon_list": ["AK-47", "Glock-18", "Knife"],
                "side": "T",
                "ct_alive": 5,
                "t_alive": 5,
                "round_time_left": 90,
                "bomb_planted": False,
                "freeze_time": False,
                "score": [0, 0],
            }
            states.append(state)

        # 8. Stop audio
        if self.audio_recorder:
            self.audio_recorder.stop_recording()

        # 9. Flush frames
        self.saver.flush()

        print(f"[Engine] Scenario {scenario_id} done: {len(states)} ticks saved")

        return {
            "scenario_id": scenario_id,
            "scenario_type": scenario.scenario_type,
            "map_name": scenario.map_name,
            "demo_path": scenario_dir,
            "audio_path": audio_dir if self.audio_recorder else None,
            "states": states,
            "delta_yaw": scenario.delta_yaw,
            "delta_pitch": scenario.delta_pitch,
            "n_bots": len([b for b in scenario.bots if not b.label.startswith("tracking_end")]),
        }

    # ------------------------------------------------------------------ #
    # Batch execution                                                      #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        scenarios: List[Scenario],
        trajectory_gen: TrajectoryGenerator,
        n_ticks: int = 64,
    ) -> List[Dict[str, Any]]:
        """Run a batch of scenarios, return metadata for all."""
        results = []

        for i, scenario in enumerate(scenarios):
            print(f"\n[Engine] === Scenario {i+1}/{len(scenarios)} ===")
            trajectory = trajectory_gen.generate(scenario, n_ticks)
            meta = self.run_scenario(scenario, trajectory)
            if meta:
                results.append(meta)

        return results

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def start_capture(self):
        """Start the WGC screen capture."""
        self.capture.start()
        time.sleep(1)
        if not self.capture.is_running:
            raise RuntimeError("Screen capture failed to start")
        print("[Engine] Screen capture started")

    def stop(self):
        """Stop capture and cleanup."""
        self.cmd.send("host_timescale 1")
        self.cmd.send("noclip 0")
        self.capture.stop()
        self.saver.shutdown()
        print("[Engine] Stopped")
