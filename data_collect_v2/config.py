"""
Configuration for CS2 Dataset Collector v2.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CollectorConfig:
    # ---- Required ----
    demo_dir: str           # Directory with .dem files
    output_dir: str         # Root output: frames/, audio/, dataset.json
    steam_id: int           # Target player Steam64 ID

    # ---- Capture ----
    cs2_window_name: str = "Counter-Strike 2"
    capture_backend: str = "wgc"        # "wgc" or "dxcam"
    target_fps: int = 64                # Capture rate = tickrate (every tick)
    frame_size: int = 640               # Output JPEG size (square)
    jpeg_quality: int = 85

    # ---- Audio ----
    record_audio: bool = True
    audio_sample_rate: int = 16000      # Output WAV sample rate

    # ---- Sandbox ----
    sandbox_name: Optional[str] = None  # Sandboxie sandbox name (--sandbox)

    # ---- Demo ----
    tickrate: int = 64
    tick_stride: int = 1                # Save every Nth tick (1 = every tick)
    skip_first_rounds: int = 3          # Skip warmup rounds
    playback_speed: float = 1.0         # Must be 1.0 for audio sync

    # ---- Performance ----
    saver_threads: int = 8              # JPEG compression threads
    saver_max_pending: int = 128        # Max queued frames before dropping
    capture_poll_hz: int = 200          # Capture loop poll rate

    # ---- Radar ----
    render_radar: bool = True               # Render radar overlay onto captured frames

    # ---- Virtual Display ----
    virtual_display: Optional[int] = None   # Monitor index to move CS2 to (None = disabled)

    # ---- Paths ----
    cs2_demo_dir: str = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"

    @property
    def console_log_path(self) -> str:
        import os
        return os.path.join(self.cs2_demo_dir, "console.log")

    @property
    def frames_dir(self) -> str:
        import os
        return os.path.join(self.output_dir, "frames")

    @property
    def audio_dir(self) -> str:
        import os
        return os.path.join(self.output_dir, "audio")

    @property
    def assets_dir(self) -> str:
        import os
        return os.path.join(os.path.dirname(__file__), "assets")

    @property
    def dataset_json_path(self) -> str:
        import os
        return os.path.join(self.output_dir, "dataset.json")
