"""
Full Dataset Collection Pipeline for CS2.

Complete pipeline to:
1. Record demo playback (video + audio at NORMAL speed)
2. Parse demo into JSON (states + events for BC and RL)
3. Synchronize audio with parsed ticks

Output structure:
    OUTPUT_DIR/
    ├── videos/
    │   └── demo_name/
    │       ├── round_0.mp4
    │       └── ...
    ├── audio/
    │   └── demo_name/
    │       ├── round_0.wav  (16kHz mono)
    │       └── ...
    ├── frames/  (optional - extract frames from video)
    │   └── demo_name/
    │       ├── tick_0000.png
    │       └── ...
    └── parsed/
        └── demo_name.json  (states + events)

Usage:
    python full_dataset_pipeline.py \
        --demo-dir "C:/path/to/demos" \
        --output-dir "D:/CS2_Dataset" \
        --steam-id 76561198386265483 \
        --record-video \
        --record-audio \
        --parse-demo

    # Or run steps separately:
    # Step 1: Record only
    python full_dataset_pipeline.py --demo-dir ... --record-video --record-audio

    # Step 2: Parse only
    python full_dataset_pipeline.py --demo-dir ... --parse-demo --output-dir ...
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional
import keyboard

# Import modules
from screen_capture_with_audio import CS2DemoRecorder
from universal_demo_parser import UniversalDemoParser


class CS2DatasetPipeline:
    """
    Full pipeline for CS2 dataset collection.
    """

    def __init__(
        self,
        demo_dir: str,
        output_dir: str,
        steam_id: int,
        skip_first_rounds: int = 3
    ):
        """
        Args:
            demo_dir: Directory containing .dem files
            output_dir: Output directory for all data
            steam_id: Target player Steam ID
            skip_first_rounds: Skip first N rounds (warmup)
        """
        self.demo_dir = demo_dir
        self.output_dir = output_dir
        self.steam_id = steam_id
        self.skip_first_rounds = skip_first_rounds

        # Create output directories
        self.video_dir = os.path.join(output_dir, "videos")
        self.audio_dir = os.path.join(output_dir, "audio")
        self.frames_dir = os.path.join(output_dir, "frames")
        self.parsed_dir = os.path.join(output_dir, "parsed")

        for d in [self.video_dir, self.audio_dir, self.frames_dir, self.parsed_dir]:
            os.makedirs(d, exist_ok=True)

    def find_demos(self) -> List[str]:
        """Find all .dem files in demo directory."""
        demos = [
            os.path.join(self.demo_dir, f)
            for f in os.listdir(self.demo_dir)
            if f.endswith('.dem')
        ]
        return sorted(demos)

    def record_demos(
        self,
        record_video: bool = True,
        record_audio: bool = True,
        start_index: int = 0
    ):
        """
        Record demo playback (video + audio).

        Args:
            record_video: Whether to record video via OBS
            record_audio: Whether to record audio via WASAPI
            start_index: Start from demo index
        """
        print("\n" + "="*60)
        print("STEP 1: Recording demos (video + audio)")
        print("="*60)
        print(f"Video: {record_video}, Audio: {record_audio}")
        print(f"Output: {self.video_dir}")
        print("Press ESCAPE to stop recording")
        print("="*60 + "\n")

        recorder = CS2DemoRecorder(
            demo_dir=self.demo_dir,
            record_dir=self.video_dir,
            steam_id=self.steam_id,
            record_audio=record_audio
        )

        try:
            recorder.run(start_index=start_index)
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        except Exception as e:
            print(f"\nError during recording: {e}")

        print("\nRecording complete!")

    def parse_demos(self):
        """
        Parse demos into JSON format (states + events).
        """
        print("\n" + "="*60)
        print("STEP 2: Parsing demos")
        print("="*60)
        print(f"Demo dir: {self.demo_dir}")
        print(f"Output: {self.parsed_dir}")
        print("="*60 + "\n")

        demos = self.find_demos()
        print(f"Found {len(demos)} demos to parse")

        parser = UniversalDemoParser(target_steam_id=self.steam_id)

        for demo_path in demos:
            demo_name = os.path.basename(demo_path)[:-4]
            output_json = os.path.join(self.parsed_dir, f"{demo_name}.json")

            # Skip if already parsed
            if os.path.exists(output_json):
                print(f"Skipping {demo_name} (already parsed)")
                continue

            try:
                print(f"\nParsing {demo_name}...")
                demo_data = parser.parse_demo(demo_path, self.skip_first_rounds)

                # Save individual demo JSON
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump({'demos': [demo_data]}, f, indent=2, ensure_ascii=False)

                print(f"  Saved {len(demo_data['rounds'])} rounds to {output_json}")

            except Exception as e:
                print(f"  Error parsing {demo_name}: {e}")

        print("\nParsing complete!")

    def create_dataset_json(self, output_name: str = "dataset.json"):
        """
        Combine all parsed demos into a single dataset JSON.

        Args:
            output_name: Output filename
        """
        print("\n" + "="*60)
        print("STEP 3: Creating combined dataset JSON")
        print("="*60)

        all_demos = []
        parsed_files = [f for f in os.listdir(self.parsed_dir) if f.endswith('.json')]

        for parsed_file in parsed_files:
            with open(os.path.join(self.parsed_dir, parsed_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_demos.extend(data['demos'])

        # Save combined dataset
        output_path = os.path.join(self.output_dir, output_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'demos': all_demos}, f, indent=2, ensure_ascii=False)

        print(f"Combined {len(all_demos)} demos into {output_path}")

        # Print statistics
        total_rounds = sum(len(demo['rounds']) for demo in all_demos)
        total_states = sum(
            len(round_data['states'])
            for demo in all_demos
            for round_data in demo['rounds']
        )
        total_events = sum(
            len(round_data['events'])
            for demo in all_demos
            for round_data in demo['rounds']
        )

        print("\nDataset statistics:")
        print(f"  Total demos: {len(all_demos)}")
        print(f"  Total rounds: {total_rounds}")
        print(f"  Total states: {total_states}")
        print(f"  Total events: {total_events}")

    def run_full_pipeline(
        self,
        record_video: bool = True,
        record_audio: bool = True,
        parse_demo: bool = True,
        start_index: int = 0
    ):
        """
        Run the complete pipeline.

        Args:
            record_video: Whether to record video
            record_audio: Whether to record audio
            parse_demo: Whether to parse demos
            start_index: Start from demo index (for recording)
        """
        print("\n" + "="*60)
        print("CS2 DATASET COLLECTION PIPELINE")
        print("="*60)
        print(f"Demo dir: {self.demo_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Steam ID: {self.steam_id}")
        print("="*60 + "\n")

        # Step 1: Record (if enabled)
        if record_video or record_audio:
            self.record_demos(record_video, record_audio, start_index)

        # Step 2: Parse (if enabled)
        if parse_demo:
            self.parse_demos()
            self.create_dataset_json()

        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Dataset Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (record + parse)
  python full_dataset_pipeline.py \\
    --demo-dir "C:/path/to/demos" \\
    --output-dir "D:/CS2_Dataset" \\
    --steam-id 76561198386265483 \\
    --record-video --record-audio --parse-demo

  # Record only
  python full_dataset_pipeline.py \\
    --demo-dir "C:/path/to/demos" \\
    --output-dir "D:/CS2_Dataset" \\
    --steam-id 76561198386265483 \\
    --record-video --record-audio

  # Parse only (after recording)
  python full_dataset_pipeline.py \\
    --demo-dir "C:/path/to/demos" \\
    --output-dir "D:/CS2_Dataset" \\
    --steam-id 76561198386265483 \\
    --parse-demo
        """
    )

    parser.add_argument(
        '--demo-dir',
        type=str,
        required=True,
        help='Directory containing .dem files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--steam-id',
        type=int,
        required=True,
        help='Target player Steam ID'
    )
    parser.add_argument(
        '--skip-rounds',
        type=int,
        default=3,
        help='Skip first N rounds (warmup)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start from demo index (for recording)'
    )
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Record video via OBS'
    )
    parser.add_argument(
        '--record-audio',
        action='store_true',
        help='Record audio via WASAPI'
    )
    parser.add_argument(
        '--parse-demo',
        action='store_true',
        help='Parse demos to JSON'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = CS2DatasetPipeline(
        demo_dir=args.demo_dir,
        output_dir=args.output_dir,
        steam_id=args.steam_id,
        skip_first_rounds=args.skip_rounds
    )

    # Run
    try:
        pipeline.run_full_pipeline(
            record_video=args.record_video,
            record_audio=args.record_audio,
            parse_demo=args.parse_demo,
            start_index=args.start_index
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
