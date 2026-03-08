"""
Single-Window Data Collector — main orchestrator.

Collects CS2 demo data: frame screenshots (every tick) + audio (per round).
Outputs in the format expected by final_model/DatasetIntent.py.
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List

import keyboard
from demoparser2 import DemoParser
from tqdm import tqdm

from config import CollectorConfig
from capture import create_capture
from audio import AudioRecorder
from demo_control import DemoController
from frame_saver import FrameSaver
from tick_clock import TickClock
from json_builder import DatasetJsonBuilder
from virtual_display import setup_virtual_display, restore_window_style, lock_cursor_to_physical, unlock_cursor

# Add universal_demo_parser to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'audio_adaptation', 'data_collect'))
from universal_demo_parser import UniversalDemoParser


def _switch_to_english_layout():
    """Switch active keyboard layout to English (US) system-wide."""
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        KLF_ACTIVATE = 0x00000001

        hkl = user32.LoadKeyboardLayoutW("00000409", KLF_ACTIVATE)

        # Post layout change to all visible top-level windows
        @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def _enum_cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                user32.PostMessageW(hwnd, 0x0050, 0, lparam)  # WM_INPUTLANGCHANGEREQUEST
            return True

        user32.EnumWindows(_enum_cb, hkl)
        print("[Collector] Keyboard layout switched to English (US)")
    except Exception as e:
        print(f"[Collector] WARNING: Could not switch keyboard layout: {e}")


class SingleWindowCollector:
    """
    Collects dataset from CS2 demo playback.

    Per demo:
      1. Parse .dem → round boundaries, tick states, events
      2. Load demo in CS2, set up spectator
      3. For each round: capture frames (every tick) + audio
      4. Save frames as tick_{N}.jpg, audio as round_{id}.wav
    """

    def __init__(self, config: CollectorConfig):
        self.config = config
        self._stop_requested = False

        # Components
        self.capture = create_capture(config.capture_backend, config.cs2_window_name)
        self.audio = AudioRecorder(
            sample_rate=config.audio_sample_rate,
        ) if config.record_audio else None
        self.demo_ctrl = DemoController(
            cs2_window_name=config.cs2_window_name,
            demo_dir=config.cs2_demo_dir,
            sandbox_name=config.sandbox_name,
        )
        self.saver = FrameSaver(
            num_threads=config.saver_threads,
            jpeg_quality=config.jpeg_quality,
            frame_size=config.frame_size,
            max_pending=config.saver_max_pending,
        )
        self.tick_clock = TickClock(
            tickrate=config.tickrate,
            playback_speed=config.playback_speed,
            tick_stride=config.tick_stride,
        )
        self.parser = UniversalDemoParser(config.steam_id)
        self.json_builder = DatasetJsonBuilder()

    def _find_user_id(self, demo_path: str, round_start_ticks: List[int]) -> Optional[int]:
        """Find the spec_player user_id for the target steam_id."""
        parser = DemoParser(demo_path)
        ticks_df = parser.parse_ticks(['user_id', 'steamid'])

        lookup_tick = round_start_ticks[0] if round_start_ticks else 300
        tmp = ticks_df[ticks_df['tick'] == lookup_tick]
        rows = tmp[tmp['steamid'] == self.config.steam_id]['user_id']

        if rows.empty:
            print(f"[Collector] Player {self.config.steam_id} not found at tick {lookup_tick}")
            return None

        return int(rows.iloc[0]) + 1  # +1 for spec_player command

    def collect_demo(self, demo_path: str) -> Optional[Dict[str, Any]]:
        """
        Collect data from a single demo file.

        Returns dict entry for dataset.json, or None on failure.
        """
        demo_name = os.path.basename(demo_path)[:-4]
        print(f"\n{'='*60}")
        print(f"Processing demo: {demo_name}")
        print(f"{'='*60}")

        # Output dirs
        frames_dir = os.path.join(self.config.frames_dir, demo_name)
        audio_dir = os.path.join(self.config.audio_dir, demo_name)
        os.makedirs(frames_dir, exist_ok=True)
        if self.config.record_audio:
            os.makedirs(audio_dir, exist_ok=True)

        # 1. Parse demo
        try:
            demo_data = self.parser.parse_demo(demo_path, self.config.skip_first_rounds)
        except Exception as e:
            print(f"[Collector] Failed to parse demo: {e}")
            return None

        rounds = demo_data['rounds']
        if not rounds:
            print("[Collector] No rounds found, skipping")
            return None

        # 2. Find user_id for spec_player
        round_start_ticks = [r['start_tick'] for r in rounds]
        user_id = self._find_user_id(demo_path, round_start_ticks)
        if user_id is None:
            return None

        # 3. Setup console log reader for tick sync
        self.demo_ctrl.setup_console_log(self.config.console_log_path)

        # 4. Load demo in CS2 via direct console commands (full path for non-csgo dirs)
        self.demo_ctrl.load_demo(demo_path, user_id)

        # 5. Parse server_start_tick for tick offset
        self.demo_ctrl.parse_server_start_tick()

        # 6. Start screen capture
        self.capture.start()
        time.sleep(1)

        if not self.capture.is_running:
            print("[Collector] Capture failed to start!")
            self.demo_ctrl.cleanup()
            return None

        # 7. Setup radar renderer once (parse positions before round loop)
        radar_renderer = None
        if self.config.render_radar:
            radar_renderer = self._setup_radar_renderer(demo_path)

        # 8. Record each round + apply radar overlay immediately after
        for round_data in tqdm(rounds, desc="Rounds"):
            if self._stop_requested:
                print("[Collector] Stop requested by user")
                break

            self._collect_round(round_data, frames_dir, audio_dir)

            # Flush saver before radar overlay AND before next round starts.
            # Without flush, pending frames fill max_pending=128 and new frames drop.
            self.saver.flush()

            if radar_renderer is not None:
                self.demo_ctrl.pause()
                self._render_radar_round(radar_renderer, round_data, frames_dir)
                self.demo_ctrl.resume()

        # 9. Cleanup
        self.capture.stop()
        self.saver.flush()
        self.saver.print_stats()
        self.demo_ctrl.stop_demo()

        # 10. Build dataset entry (rounds now contain drift_info)
        entry = {
            'demo_path': frames_dir,
            'audio_path': audio_dir,
            'server_start_tick': self.demo_ctrl._server_start_tick,
            'rounds': rounds,
        }
        return entry

    def _collect_round(self, round_data: Dict, frames_dir: str, audio_dir: str):
        """Capture frames and audio for a single round."""
        round_id = round_data['round_id']
        start_tick = round_data['start_tick']
        end_tick = round_data['end_tick']

        # Find player death tick — stop capturing after death
        death_tick = None
        for event in round_data.get('events', []):
            if event['type'] == 'death':
                death_tick = event['tick']
                break

        effective_end = end_tick
        if death_tick is not None and death_tick < end_tick:
            ticks_saved = end_tick - death_tick
            seconds_saved = ticks_saved / self.config.tickrate
            print(f"    Player died at tick {death_tick}, "
                  f"skipping {ticks_saved} ticks ({seconds_saved:.1f}s)")
            effective_end = death_tick

        duration_sec = (effective_end - start_tick) / self.config.tickrate
        print(f"  Round {round_id}: ticks {start_tick}-{effective_end} ({duration_sec:.0f}s)")

        # Seek to round start — pauses demo and marks log position
        # CS2 writes "paused on tick X" to console.log after seek
        self.demo_ctrl.goto_tick_and_pause(start_tick)

        # --- Calibrate tick clock at round start ---
        # Read the tick CS2 wrote after seeking (no extra resume/pause needed)
        real_start = self.demo_ctrl.read_paused_tick()
        if real_start is not None:
            print(f"    Calibrated start: real_tick={real_start} (expected={start_tick}, "
                  f"delta={real_start - start_tick})")
            calibrated_start = real_start
        else:
            print("    WARNING: tick calibration failed, using expected start_tick")
            calibrated_start = start_tick

        # Resume playback
        self.demo_ctrl.resume()
        time.sleep(0.1)

        # Start audio recording (after resume so we don't record silence)
        if self.audio:
            audio_path = os.path.join(audio_dir, f"round_{round_id}.wav")
            self.audio.start_recording(audio_path)

        # Start tick clock with calibrated tick
        self.tick_clock.start_round(calibrated_start)

        # Capture loop
        last_saved_tick = calibrated_start - 1
        poll_interval = 1.0 / self.config.capture_poll_hz  # ~5ms at 200Hz
        frames_saved = 0
        last_stale_check = time.time()

        while not self._stop_requested:
            current_tick = self.tick_clock.current_tick()
            if current_tick >= effective_end:
                break

            # Periodically check if CS2 got minimized (every 2 sec)
            now = time.time()
            if now - last_stale_check > 2.0:
                last_stale_check = now
                if self.capture.is_stale(timeout=2.0):
                    self.demo_ctrl.ensure_not_minimized()

            # Check if we should save a new frame
            new_tick = self.tick_clock.should_capture(last_saved_tick)
            if new_tick is not None and new_tick < effective_end:
                frame = self.capture.get_frame()
                if frame is not None:
                    self.saver.save(frame, new_tick, frames_dir)
                    last_saved_tick = new_tick
                    frames_saved += 1
                # Don't sleep if behind — catch up immediately
                if self.tick_clock.current_tick() > last_saved_tick + self.config.tick_stride:
                    continue
            else:
                time.sleep(poll_interval)

        # Stop audio recording before pause (capture real audio until end)
        if self.audio:
            self.audio.stop_recording()

        # --- Measure drift at round end ---
        real_end = self.demo_ctrl.pause_and_read_tick()
        if real_end is not None:
            self.tick_clock.record_end_tick(real_end)
        self.demo_ctrl.resume()

        self.tick_clock.stop()

        # Save drift info and coverage into round_data for dataset.json
        round_data['drift_info'] = self.tick_clock.drift_info

        expected_frames = (effective_end - calibrated_start) // self.config.tick_stride
        actual = frames_saved
        coverage = actual / max(expected_frames, 1) * 100
        round_data['frame_coverage'] = {
            'frames_saved': actual,
            'frames_expected': expected_frames,
            'coverage_pct': round(coverage, 1),
        }
        print(f"    Frames: {actual}/{expected_frames} ({coverage:.0f}% coverage)")

    def collect_all(self, resume: bool = False, limit: int = 0, offset: int = 0):
        """
        Process all demos in demo_dir.

        Args:
            resume: If True, skip demos that already have frames_dir.
            limit:  Process only N demos after offset (0 = all remaining).
            offset: Skip first N demos — use to split work across VMs.
                    VM1: offset=0, limit=50  → demos[0:50]
                    VM2: offset=50, limit=0  → demos[50:]
        """
        # Switch keyboard layout to English (US) — required for keyboard library
        # and to avoid any layout-dependent input issues
        _switch_to_english_layout()

        # Move CS2 to virtual display if configured
        self._using_virtual_display = False
        if self.config.virtual_display is not None:
            # -1 = auto-detect, >=0 = specific monitor index
            monitor_idx = None if self.config.virtual_display == -1 else self.config.virtual_display
            if setup_virtual_display(self.config.cs2_window_name, monitor_idx):
                self._using_virtual_display = True
                lock_cursor_to_physical(monitor_idx)
                print("[Collector] CS2 moved to virtual display — you can work freely")
            else:
                print("[Collector] WARNING: Virtual display setup failed, continuing on primary monitor")

        # Connect to CS2 (find window)
        print("[Collector] Connecting to CS2...")
        if not self.demo_ctrl.connect():
            print("[Collector] FATAL: Cannot find CS2 window. Is CS2 running?")
            return

        demos = sorted([
            f for f in os.listdir(self.config.demo_dir)
            if f.endswith('.dem')
        ])
        if offset > 0:
            demos = demos[offset:]
        if limit > 0:
            demos = demos[:limit]
        print(f"Found {len(demos)} demos to process")

        # Setup Escape listener
        keyboard.on_press_key('esc', lambda _: self._on_escape())
        print("Press ESCAPE at any time to stop recording")

        all_entries = []

        for i, demo_file in enumerate(demos):
            if self._stop_requested:
                break

            demo_path = os.path.join(self.config.demo_dir, demo_file)
            demo_name = demo_file[:-4]

            # Resume: skip if already processed
            if resume:
                frames_dir = os.path.join(self.config.frames_dir, demo_name)
                if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 100:
                    print(f"[{i+1}/{len(demos)}] Skipping {demo_name} (already exists)")
                    continue

            print(f"\n[{i+1}/{len(demos)}] {demo_name}")
            entry = self.collect_demo(demo_path)

            if entry:
                all_entries.append(entry)
                # Save dataset.json incrementally
                self.json_builder.save(all_entries, self.config.dataset_json_path)

        # Cleanup
        keyboard.unhook_all()
        self.saver.shutdown()
        self.demo_ctrl.cleanup()

        # Restore window and cursor if virtual display was used
        if self._using_virtual_display:
            unlock_cursor()
            restore_window_style(self.config.cs2_window_name)
            print("[Collector] Restored CS2 window and cursor")

        print(f"\nDone! Processed {len(all_entries)} demos.")
        print(f"Dataset saved to: {self.config.dataset_json_path}")

    def _setup_radar_renderer(self, demo_path: str):
        """Parse positions + create RadarRenderer once per demo. Returns renderer or None."""
        from demo_parser_positions import parse_positions
        from radar_renderer import RadarRenderer

        print("[Collector] Parsing positions for radar overlay...")

        try:
            positions_df = parse_positions(demo_path)
        except Exception as e:
            print(f"[Collector] Failed to parse positions: {e}")
            return None

        try:
            header = DemoParser(demo_path).parse_header()
            map_name = header.get('map_name', '')
        except Exception:
            map_name = ''

        if not map_name:
            print("[Collector] WARNING: Could not detect map name, skipping radar")
            return None

        try:
            renderer = RadarRenderer(map_name, self.config.steam_id, self.config.assets_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"[Collector] Radar not available for map '{map_name}': {e}")
            return None

        # Attach positions_df to renderer so _render_radar_round can use it
        renderer._positions_df = positions_df
        print(f"[Collector] Radar renderer ready for {map_name}")
        return renderer

    def _render_radar_round(self, renderer, round_data: Dict, frames_dir: str):
        """Apply radar overlay to all saved frames of a single round.

        Steps:
          1. Glob existing tick_*.jpg — avoids per-tick os.path.exists calls
          2. Pre-render all radar images sequentially (numpy, order-dependent)
          3. Write frames in parallel via ThreadPoolExecutor (I/O bound)
        """
        import glob as _glob
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        start_tick = round_data['start_tick']
        end_tick = round_data['end_tick']

        renderer.setup_round(renderer._positions_df, start_tick)

        # Collect existing frame paths for this round
        pattern = os.path.join(frames_dir, "tick_*.jpg")
        all_paths = _glob.glob(pattern)
        round_frames = []
        for p in all_paths:
            try:
                tick = int(os.path.basename(p)[5:-4])  # "tick_12345.jpg" → 12345
                if start_tick <= tick <= end_tick:
                    round_frames.append((tick, p))
            except ValueError:
                continue
        round_frames.sort()

        if not round_frames:
            print(f"    Radar overlay: 0 frames")
            return

        # Pre-render all radar images (sequential — renderer has per-tick state)
        jpeg_quality = self.config.jpeg_quality
        rendered = []
        for tick, path in round_frames:
            radar_img = renderer.render_tick(tick)
            radar_resized = cv2.resize(radar_img, (135, 185))
            rendered.append((path, radar_resized))

        # Parallel write (I/O bound)
        def _write(args):
            path, radar = args
            frame = cv2.imread(path)
            if frame is None:
                return False
            frame[5:190, 0:135] = radar
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            return True

        workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_write, rendered))

        print(f"    Radar overlay: {sum(results)}/{len(rendered)} frames")

    def _on_escape(self):
        print("\n[ESCAPE] Stop requested!")
        self._stop_requested = True
