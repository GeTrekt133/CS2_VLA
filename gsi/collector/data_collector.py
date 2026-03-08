"""
Data Collector - collects game data via GSI for training.

Synchronizes:
- Game state from GSI
- Screenshots from screen capture
- Saves in train_dataset.json format
"""
import os
import json
import time
import threading
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

from ..server.http_server import GSIServer, GSIPayload
from ..adapter.state_adapter import GSIStateAdapter, GameStateSnapshot
from ..buffer.state_buffer import StateBuffer, FrameBuffer
from ..config.gsi_config import GSIConfig


class RoundData:
    """Data for a single round."""

    def __init__(self, round_id: int, start_tick: int):
        self.round_id = round_id
        self.start_tick = start_tick
        self.end_tick: Optional[int] = None
        self.states: List[Dict] = []

    def to_dict(self) -> Dict:
        """Convert to dict format."""
        return {
            "round_id": self.round_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick or self.start_tick,
            "states": self.states,
        }


class GSIDataCollector:
    """
    Collects CS2 game data via GSI for training.

    Synchronizes:
    - Game state from GSI
    - Screenshots from screen capture
    - Saves in train_dataset.json format

    Usage:
        collector = GSIDataCollector(
            config=GSIConfig(output_dir="./collected_data")
        )
        collector.start()
        # ... play CS2 ...
        collector.stop()
        collector.save()
    """

    def __init__(self, config: Optional[GSIConfig] = None):
        self.config = config or GSIConfig()

        # Components
        self.server = GSIServer(
            host=self.config.host,
            port=self.config.port,
            auth_token=self.config.auth_token,
        )
        self.adapter = GSIStateAdapter()
        self.state_buffer = StateBuffer(size=self.config.state_buffer_size)
        self.frame_buffer = FrameBuffer(size=self.config.frame_buffer_size)

        # Collection state
        self._collecting = False
        self._current_game_id: Optional[str] = None
        self._current_round: Optional[RoundData] = None
        self._rounds: List[RoundData] = []
        self._last_round_num: int = 0
        self._tick_counter: int = 0

        # Screen capture
        self._capture_thread: Optional[threading.Thread] = None
        self._sct = None  # Lazy init

        # Output paths
        self._session_dir: Optional[Path] = None

        # Stats
        self._states_collected = 0
        self._frames_collected = 0

    def _init_screen_capture(self):
        """Initialize screen capture (mss)."""
        try:
            import mss
            self._sct = mss.mss()
            return True
        except ImportError:
            print("[Collector] WARNING: mss not installed, screenshots disabled")
            print("[Collector] Install with: pip install mss")
            return False

    def start(self):
        """Start data collection."""
        if self._collecting:
            print("[Collector] Already collecting")
            return

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_game_id = f"gsi_session_{timestamp}"
        self._session_dir = Path(self.config.output_dir) / self._current_game_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Reset state
        self._rounds = []
        self._current_round = None
        self._last_round_num = 0
        self._tick_counter = 0
        self._states_collected = 0
        self._frames_collected = 0

        # Register GSI callback
        self.server.register_callback(self._on_gsi_update)
        self.server.start()

        # Start screen capture thread
        self._collecting = True
        if self.config.save_screenshots:
            if self._init_screen_capture():
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    daemon=True
                )
                self._capture_thread.start()

        print(f"[Collector] Started collecting to {self._session_dir}")
        print(f"[Collector] Waiting for CS2 GSI data on port {self.config.port}...")

    def _on_gsi_update(self, payload: GSIPayload):
        """Handle incoming GSI data."""
        if not self._collecting:
            return

        # Parse to snapshot
        snapshot = self.adapter.parse(payload)
        if snapshot is None:
            return  # Not in-game

        # Detect round changes
        if snapshot.round_num != self._last_round_num:
            self._on_round_change(snapshot.round_num, self._tick_counter)
            self._last_round_num = snapshot.round_num

        # Skip if no active round or in menu
        if self._current_round is None:
            return

        # Convert to state_vec
        state_vec = self.adapter.to_state_vec(snapshot)

        # Get synchronized frame
        frame = None
        if self.config.save_screenshots:
            frame_data = self.frame_buffer.get_closest(snapshot.timestamp)
            if frame_data:
                _, frame = frame_data

        # Add to buffer
        self.state_buffer.add(snapshot, state_vec, frame=frame)

        # Record state (every screenshot_interval ticks)
        if self._tick_counter % self.config.screenshot_interval == 0:
            self._record_state(snapshot, self._tick_counter, frame)

        self._tick_counter += 1

    def _on_round_change(self, new_round: int, current_tick: int):
        """Handle round transition."""
        # End current round
        if self._current_round is not None:
            self._current_round.end_tick = current_tick - 1
            if self._current_round.states:  # Only save non-empty rounds
                self._rounds.append(self._current_round)
            print(f"[Collector] Round {self._current_round.round_id} ended "
                  f"({len(self._current_round.states)} states)")

        # Start new round
        self._current_round = RoundData(
            round_id=new_round,
            start_tick=current_tick
        )
        print(f"[Collector] Round {new_round} started at tick {current_tick}")

    def _record_state(
        self,
        snapshot: GameStateSnapshot,
        tick: int,
        frame: Optional[np.ndarray] = None
    ):
        """Record a single game state."""
        if self._current_round is None:
            return

        # Save frame if available
        frame_path = None
        if frame is not None and self.config.save_screenshots:
            frame_path = self._save_frame(frame, tick)
            self._frames_collected += 1

        # Build state dict matching train_dataset.json format
        state_dict = {
            "tick": tick,
            "keys": [],  # GSI doesn't provide raw key presses
            "mouse": [0.0, 0.0],  # GSI doesn't provide raw mouse data
            "detects": [],
            "hp": snapshot.hp,
            "armor": snapshot.armor,
            "helmet": snapshot.helmet,
            "defuser": False,  # Not available in basic GSI
            "weapon": snapshot.weapon,
            "ammo": snapshot.ammo,
            "round_time_left": snapshot.round_time_left,
            "bomb_planted": snapshot.bomb_planted,
            "freeze_time": snapshot.freeze_time,
            "score": [snapshot.score_t, snapshot.score_ct] if snapshot.side == "T" else [snapshot.score_ct, snapshot.score_t],
            "side": snapshot.side,
            "ct_alive": snapshot.ct_alive,
            "t_alive": snapshot.t_alive,
            "weapon_list": snapshot.weapon_list,
        }

        if frame_path:
            state_dict["frame_path"] = frame_path

        self._current_round.states.append(state_dict)
        self._states_collected += 1

    def _capture_loop(self):
        """Screen capture loop."""
        if self._sct is None:
            return

        monitor = self._sct.monitors[1]  # Primary monitor

        target_interval = 1.0 / self.config.capture_fps

        while self._collecting:
            try:
                start = time.time()

                # Capture screen
                screenshot = self._sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

                # Resize to target size
                frame = cv2.resize(
                    frame,
                    (self.config.screen_width, self.config.screen_height)
                )

                # Add to buffer with timestamp
                self.frame_buffer.add(time.time(), frame)

                # Maintain target FPS
                elapsed = time.time() - start
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"[Collector] Capture error: {e}")
                time.sleep(0.1)

    def _save_frame(self, frame: np.ndarray, tick: int) -> str:
        """Save frame to disk."""
        frame_filename = f"tick_{tick:06d}.jpg"
        frame_path = self._session_dir / frame_filename

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return frame_filename  # Return relative path

    def stop(self):
        """Stop data collection."""
        if not self._collecting:
            return

        self._collecting = False

        # End current round
        if self._current_round is not None:
            self._current_round.end_tick = self._tick_counter
            if self._current_round.states:
                self._rounds.append(self._current_round)

        # Stop server
        self.server.stop()

        # Wait for capture thread
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        print(f"[Collector] Stopped. Collected {len(self._rounds)} rounds, "
              f"{self._states_collected} states, {self._frames_collected} frames")

    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save collected data to JSON format.

        Returns path to saved file.
        """
        if not output_path:
            output_path = str(self._session_dir / "dataset.json")

        # Build dataset structure matching train_dataset.json
        dataset = {
            "demos": [
                {
                    "demo_path": str(self._session_dir),
                    "rounds": [rnd.to_dict() for rnd in self._rounds]
                }
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"[Collector] Saved dataset to {output_path}")
        return output_path

    @property
    def stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "rounds": len(self._rounds),
            "states": self._states_collected,
            "frames": self._frames_collected,
            "current_round": self._current_round.round_id if self._current_round else None,
            "tick": self._tick_counter,
        }
