"""
Screen and Audio Capture for CS2 Demo Recording.

Enhanced version of screen_capture.py with synchronized audio recording.
Records both video (via OBS) and audio (via WASAPI loopback) for each round.

Output structure:
    RECORD_DIR/
    ├── demo_name/
    │   ├── round_0.mp4  (video from OBS)
    │   ├── round_1.mp4
    │   └── ...
    └── audio/
        └── demo_name/
            ├── round_0.wav  (16kHz mono)
            ├── round_1.wav
            └── ...
"""

import obsws_python as obs
from demoparser2 import DemoParser
import subprocess
import time
import os
import pyautogui
import threading
from tqdm import tqdm
from typing import Optional
import keyboard  # For Escape listener

# Import audio recorder
from audio_capture import AudioRecorder, AudioConfig


class CS2DemoRecorder:
    """
    Synchronized video and audio recorder for CS2 demos.
    """

    def __init__(
        self,
        obs_path: str = r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
        demo_dir: str = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo",
        record_dir: str = r"D:\RecordDemos",
        websocket_host: str = "localhost",
        websocket_port: int = 4455,
        websocket_password: str = "secret",
        steam_id: int = 76561198386265483,
        record_audio: bool = True
    ):
        self.obs_path = obs_path
        self.obs_dir = os.path.dirname(obs_path)
        self.demo_dir = demo_dir
        self.record_dir = record_dir
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.websocket_password = websocket_password
        self.steam_id = steam_id
        self.record_audio = record_audio

        # Audio recorder
        self.audio_recorder: Optional[AudioRecorder] = None
        self.audio_dir = os.path.join(record_dir, "audio")

        # OBS client
        self.obs_client: Optional[obs.ReqClient] = None

        # F1 spam thread control
        self._spam_active = False
        self._spam_thread: Optional[threading.Thread] = None

        # Escape listener control
        self._stop_requested = False

    def _spam_f1(self):
        """Background thread to spam F1 key."""
        while self._spam_active:
            pyautogui.press('f1')
            time.sleep(1)

    def _start_f1_spam(self):
        """Start F1 spam thread."""
        self._spam_active = True
        self._spam_thread = threading.Thread(target=self._spam_f1, daemon=True)
        self._spam_thread.start()

    def _stop_f1_spam(self):
        """Stop F1 spam thread."""
        self._spam_active = False
        if self._spam_thread:
            self._spam_thread.join(timeout=2.0)

    def start_obs(self):
        """Start OBS in background."""
        subprocess.Popen(
            [self.obs_path, "--minimize-to-tray"],
            cwd=self.obs_dir
        )
        print("OBS started, waiting for initialization...")
        time.sleep(7)

    def connect_obs(self) -> bool:
        """Connect to OBS WebSocket."""
        try:
            self.obs_client = obs.ReqClient(
                host=self.websocket_host,
                port=self.websocket_port,
                password=self.websocket_password
            )
            print("Connected to OBS WebSocket")
            return True
        except Exception as e:
            print(f"Failed to connect to OBS: {e}")
            return False

    def setup_audio_recorder(self, demo_name: str):
        """Setup audio recorder for a demo."""
        if not self.record_audio:
            return

        demo_audio_dir = os.path.join(self.audio_dir, demo_name)
        os.makedirs(demo_audio_dir, exist_ok=True)

        self.audio_recorder = AudioRecorder(
            output_dir=demo_audio_dir,
            config=AudioConfig(
                sample_rate=16000,
                channels=1
            )
        )

    def load_demo(self, demo_path: str, user_id: int):
        """Load demo and setup camera."""
        pyautogui.press("`")
        time.sleep(0.5)

        demo_name = os.path.basename(demo_path)
        pyautogui.typewrite(f'playdemo {demo_name}\n')
        time.sleep(15)

        pyautogui.press("`")
        time.sleep(0.5)
        pyautogui.press("`")
        time.sleep(0.5)

        # Move mouse out of the way
        x, y = pyautogui.position()
        pyautogui.moveTo(x, y - 1000)
        time.sleep(0.5)

        # Setup demo UI (NO TIMESCALE - record at normal speed)
        pyautogui.typewrite('demoui\n')
        time.sleep(0.5)

        # IMPORTANT: Do NOT use demo_timescale for audio synchronization
        # pyautogui.typewrite('demo_timescale 1\n')  # Already default
        # time.sleep(0.5)

        # Bind F1 to follow player
        pyautogui.typewrite(f'bind "F1" "spec_player {user_id}"\n')
        time.sleep(0.5)

        # Start F1 spam
        self._start_f1_spam()

        # Disable X-ray
        pyautogui.typewrite('spec_show_xray 0\n')
        time.sleep(1)

        pyautogui.press("`")
        time.sleep(0.5)

    def record_round(
        self,
        round_id: int,
        start_tick: int,
        duration_seconds: float,
        save_dir: str
    ):
        """Record a single round (video + audio)."""
        # Pause and goto tick
        pyautogui.press("`")
        time.sleep(0.5)

        pyautogui.typewrite('demo_pause\n')
        time.sleep(0.5)

        pyautogui.typewrite(f'demo_gototick {start_tick}\n')
        time.sleep(3)  # Wait for seek to complete (large demos need more time)

        pyautogui.press("`")
        time.sleep(0.5)

        # Resume playback FIRST, then start recording
        # This avoids recording silence/static frame at the beginning
        pyautogui.press('f5')
        time.sleep(0.3)  # Small delay to let playback start

        # Start recording (video + audio) AFTER playback is running
        if self.record_audio and self.audio_recorder:
            self.audio_recorder.start_recording(f"round_{round_id}.wav")

        self.obs_client.start_record()

        # Wait for round duration
        time.sleep(duration_seconds)

        # Stop recording
        self.obs_client.stop_record()

        if self.record_audio and self.audio_recorder:
            audio_path = self.audio_recorder.stop_recording()
            if audio_path:
                print(f"  Audio saved: {audio_path}")

        print(f"  Round {round_id} recorded ({duration_seconds:.1f}s)")

    def _check_stop_requested(self):
        """Check if user pressed Escape to stop."""
        return self._stop_requested

    def _on_escape_press(self):
        """Handle Escape key press."""
        print("\n[ESCAPE] Stop requested by user!")
        self._stop_requested = True
        self._stop_f1_spam()

    def process_demo(self, demo_path: str, skip_first_rounds: int = 3):
        """Process a single demo file."""
        demo_name = os.path.basename(demo_path)[:-4]  # Remove .dem
        print(f"\nProcessing demo: {demo_name}")

        # Reset stop flag
        self._stop_requested = False

        try:
            # Parse demo
            parser = DemoParser(demo_path)
            start_df = parser.parse_event('round_prestart')
            round_start_ticks = list(start_df['tick'][skip_first_rounds:])

            if len(round_start_ticks) < 10:
                print(f"  Skipping: too few rounds ({len(round_start_ticks)})")
                return False

            ticks_df = parser.parse_ticks(['user_id', 'steamid'])
            max_tick = int(ticks_df.iloc[-1]['tick'])

            # Find user ID (use first round start tick where player definitely exists)
            lookup_tick = round_start_ticks[0] if round_start_ticks else 300
            tmp_df = ticks_df[ticks_df['tick'] == lookup_tick]
            player_rows = tmp_df[tmp_df['steamid'] == self.steam_id]['user_id']
            if player_rows.empty:
                print(f"  Player {self.steam_id} not found at tick {lookup_tick}")
                return False
            user_id = int(player_rows.iloc[0]) + 1

            # Setup directories
            video_save_dir = os.path.join(self.record_dir, demo_name)
            os.makedirs(video_save_dir, exist_ok=True)
            self.obs_client.set_record_directory(video_save_dir)

            # Setup audio recorder
            self.setup_audio_recorder(demo_name)

            # Load demo
            self.load_demo(demo_path, user_id)

            # Record each round
            for i, start_tick in enumerate(round_start_ticks):
                # Check if stop requested
                if self._check_stop_requested():
                    print(f"  Stopping at round {i} (user request)")
                    break

                # Calculate round duration (at normal speed, NO timescale)
                if i == len(round_start_ticks) - 1:
                    round_dur = max((max_tick - start_tick) // 64 - 25, 5)  # 64 tickrate, not 256
                else:
                    round_dur = (round_start_ticks[i + 1] - start_tick) // 64  # 64 tickrate

                print(f"  Recording round {i} (duration: {round_dur}s)...")
                self.record_round(i, start_tick, round_dur, video_save_dir)

            # Stop F1 spam
            self._stop_f1_spam()

            return True

        except Exception as e:
            print(f"  Error processing demo: {e}")
            self._stop_f1_spam()
            return False

    def run(self, start_index: int = 0):
        """Run the recorder on all demos."""
        # Find demos
        demos = [f for f in os.listdir(self.demo_dir) if f.endswith('.dem')]
        print(f"Found {len(demos)} demos")

        # Setup Escape listener
        keyboard.on_press_key('esc', lambda _: self._on_escape_press())
        print("Press ESCAPE at any time to stop recording")

        # Start OBS
        self.start_obs()

        # Connect to OBS
        if not self.connect_obs():
            return

        # Process each demo
        for i, demo in tqdm(enumerate(demos), total=len(demos)):
            if i < start_index:
                continue

            # Check if stop requested
            if self._check_stop_requested():
                print(f"\nStopping at demo {i} (user request)")
                break

            demo_path = os.path.join(self.demo_dir, demo)
            success = self.process_demo(demo_path)

            if not success:
                print(f"  Failed to process {demo}")

        # Cleanup
        keyboard.unhook_all()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CS2 Demo Recorder with Audio")
    parser.add_argument('--no-audio', action='store_true', help='Disable audio recording')
    parser.add_argument('--start-index', type=int, default=0, help='Start from demo index')
    parser.add_argument('--steam-id', type=int, default=76561198386265483, help='Steam ID')

    args = parser.parse_args()

    recorder = CS2DemoRecorder(
        record_audio=not args.no_audio,
        steam_id=args.steam_id
    )

    try:
        recorder.run(start_index=args.start_index)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
