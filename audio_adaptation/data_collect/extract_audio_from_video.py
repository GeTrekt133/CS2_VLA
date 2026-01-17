"""
Extract Audio from Existing Video Files.

Converts recorded video files (from OBS) to audio files for training.
Useful if you already have video recordings without separate audio.

Usage:
    # Extract from single video
    python extract_audio_from_video.py --input round_1.mp4 --output round_1.wav

    # Process entire demo folder
    python extract_audio_from_video.py --input-dir D:/RecordDemos/demo_name --output-dir D:/AudioData/demo_name

    # Process all demos
    python extract_audio_from_video.py --root D:/RecordDemos --audio-root D:/AudioData

Requirements:
    pip install ffmpeg-python
    # Also needs ffmpeg installed on system (https://ffmpeg.org/download.html)
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import json


# Audio extraction settings
TARGET_SAMPLE_RATE = 16000  # 16kHz for model
TARGET_CHANNELS = 1         # Mono


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            [r"C:\Users\misas\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe", '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    overwrite: bool = False
) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
        sample_rate: Target sample rate (default 16000)
        channels: Number of channels (default 1 for mono)
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    if os.path.exists(audio_path) and not overwrite:
        print(f"  Skipping (exists): {audio_path}")
        return True

    # Create output directory if needed
    audio_dir = os.path.dirname(audio_path)
    if audio_dir:
        os.makedirs(audio_dir, exist_ok=True)

    # FFmpeg command
    cmd = [
        r'C:\Users\misas\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe',
        '-i', video_path,
        '-vn',                          # No video
        '-acodec', 'pcm_f32le',         # Float32 WAV
        '-ar', str(sample_rate),        # Sample rate
        '-ac', str(channels),           # Channels
        '-y' if overwrite else '-n',    # Overwrite flag
        audio_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return True
        else:
            print(f"  Error: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"  Exception: {e}")
        return False


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


def process_demo_folder(
    demo_dir: str,
    output_dir: str,
    overwrite: bool = False
) -> dict:
    """
    Process all video files in a demo folder.

    Args:
        demo_dir: Directory with round videos (any naming convention)
        output_dir: Output directory for audio files
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with processing results
    """
    results = {
        "demo": os.path.basename(demo_dir),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "rounds": []
    }

    # Find video files (any naming convention)
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    video_files = []

    for f in os.listdir(demo_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in video_extensions:
            video_files.append(f)

    # Sort by name (works for both "round_X" and timestamp formats)
    video_files.sort()

    if not video_files:
        print(f"  No video files found in {demo_dir}")
        return results

    os.makedirs(output_dir, exist_ok=True)

    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(demo_dir, video_file)
        # Output as round_0.wav, round_1.wav, etc.
        audio_file = f"round_{idx}.wav"
        audio_path = os.path.join(output_dir, audio_file)

        # Round ID is the index in sorted order
        round_id = idx
        duration = get_video_duration(video_path)

        print(f"  Processing {video_file}...", end=" ")

        if os.path.exists(audio_path) and not overwrite:
            print("skipped (exists)")
            results["skipped"] += 1
            results["rounds"].append({
                "round_id": round_id,
                "audio_path": audio_path,
                "duration": duration,
                "status": "skipped"
            })
            continue

        success = extract_audio(video_path, audio_path, overwrite=overwrite)

        if success:
            print("done")
            results["processed"] += 1
            results["rounds"].append({
                "round_id": round_id,
                "audio_path": audio_path,
                "duration": duration,
                "status": "processed"
            })
        else:
            print("FAILED")
            results["failed"] += 1
            results["rounds"].append({
                "round_id": round_id,
                "status": "failed"
            })

    return results


def process_all_demos(
    root_dir: str,
    audio_root: str,
    overwrite: bool = False,
    save_manifest: bool = True
) -> List[dict]:
    """
    Process all demo folders in root directory.

    Args:
        root_dir: Root directory containing demo folders
        audio_root: Root directory for audio output
        overwrite: Whether to overwrite existing files
        save_manifest: Whether to save processing manifest

    Returns:
        List of processing results for each demo
    """
    all_results = []

    # Find demo folders
    demo_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item != 'audio':
            # Check if it has video files
            has_videos = any(
                f.endswith(('.mp4', '.mkv', '.avi'))
                for f in os.listdir(item_path)
            )
            if has_videos:
                demo_folders.append(item)

    print(f"Found {len(demo_folders)} demo folders")

    for demo_name in tqdm(demo_folders, desc="Processing demos"):
        demo_dir = os.path.join(root_dir, demo_name)
        output_dir = os.path.join(audio_root, demo_name)

        print(f"\nProcessing: {demo_name}")
        results = process_demo_folder(demo_dir, output_dir, overwrite)
        all_results.append(results)

    # Save manifest
    if save_manifest:
        manifest_path = os.path.join(audio_root, "audio_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")

    # Summary
    total_processed = sum(r["processed"] for r in all_results)
    total_skipped = sum(r["skipped"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)

    print(f"\n=== Summary ===")
    print(f"Processed: {total_processed}")
    print(f"Skipped: {total_skipped}")
    print(f"Failed: {total_failed}")

    return all_results


def update_dataset_json(
    dataset_json: str,
    audio_root: str,
    output_json: Optional[str] = None
):
    """
    Update existing dataset JSON with audio paths.

    Args:
        dataset_json: Path to existing dataset JSON
        audio_root: Root directory containing audio files
        output_json: Output path (default: overwrite input)
    """
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    updated_count = 0

    for demo in data.get("demos", []):
        demo_path = demo.get("demo_path", "")
        demo_name = os.path.basename(demo_path)
        audio_dir = os.path.join(audio_root, demo_name)

        if os.path.isdir(audio_dir):
            demo["audio_path"] = audio_dir
            updated_count += 1

            # Verify round audio files exist
            for rnd in demo.get("rounds", []):
                round_id = rnd.get("round_id", 0)
                round_audio = os.path.join(audio_dir, f"round_{round_id}.wav")
                if os.path.exists(round_audio):
                    rnd["audio_file"] = round_audio

    output_path = output_json or dataset_json
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated {updated_count} demos with audio paths")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio from CS2 demo videos")

    # Input modes
    parser.add_argument('--input', type=str, help='Single video file to process')
    parser.add_argument('--input-dir', type=str, help='Demo folder to process')
    parser.add_argument('--root', type=str, help='Root directory with all demos')

    # Output
    parser.add_argument('--output', type=str, help='Output audio file (for single file)')
    parser.add_argument('--output-dir', type=str, help='Output directory for audio')
    parser.add_argument('--audio-root', type=str, help='Root audio directory (for batch)')

    # Options
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')

    # Dataset update
    parser.add_argument('--update-dataset', type=str, help='Update dataset JSON with audio paths')

    args = parser.parse_args()

    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        print("  Windows: winget install ffmpeg")
        print("  Linux: apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return

    # Single file mode
    if args.input and args.output:
        print(f"Extracting audio from: {args.input}")
        success = extract_audio(
            args.input, args.output,
            sample_rate=args.sample_rate,
            overwrite=args.overwrite
        )
        if success:
            print(f"Audio saved to: {args.output}")
        return

    # Demo folder mode
    if args.input_dir and args.output_dir:
        results = process_demo_folder(
            args.input_dir,
            args.output_dir,
            overwrite=args.overwrite
        )
        print(f"\nProcessed: {results['processed']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
        return

    # Batch mode
    if args.root and args.audio_root:
        process_all_demos(
            args.root,
            args.audio_root,
            overwrite=args.overwrite,
            save_manifest=False
        )
        return

    # Update dataset
    if args.update_dataset and args.audio_root:
        update_dataset_json(args.update_dataset, args.audio_root)
        return

    # No valid mode
    parser.print_help()


if __name__ == "__main__":
    main()
