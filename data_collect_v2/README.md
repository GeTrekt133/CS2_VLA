# CS2 Dataset Collector v2

Direct frame capture + WASAPI audio recording from CS2 demo playback.
Outputs data in the format expected by `final_model/DatasetIntent.py`.

## What it does

1. Parses `.dem` files (round boundaries, player states, game events)
2. Loads each demo in CS2, spectates the target player
3. Captures every tick as a 640x640 JPEG screenshot (via Windows Graphics Capture)
4. Records round audio via WASAPI loopback (48kHz stereo -> 16kHz mono WAV)
5. Saves `dataset.json` with states, events, and paths to frames/audio

## Output structure

```
output_dir/
├── frames/
│   └── demo_name/
│       ├── tick_4568.jpg      # 640x640 JPEG
│       ├── tick_4569.jpg      # every tick (stride=1)
│       └── ...
├── audio/
│   └── demo_name/
│       ├── round_0.wav        # 16kHz mono float32
│       ├── round_1.wav
│       └── ...
└── dataset.json
```

## Requirements

**System:**
- Windows 10/11
- CS2 installed
- Python 3.10+

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Only `windows-capture` is likely not already installed. Everything else (`opencv-python`, `numpy`, `sounddevice`, `demoparser2`, etc.) should already be present from the main project.

## Usage

### Basic collection
```bash
python run.py ^
  --demo-dir "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo" ^
  --output-dir "D:/dataset" ^
  --steam-id 76561198386265483
```

### Resume (skip already processed demos)
```bash
python run.py --demo-dir "..." --output-dir "..." --resume
```

### List audio devices
```bash
python run.py --list-devices
```

### Validate dataset
```bash
python run.py --validate "D:/dataset/dataset.json"
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--demo-dir` | required | Directory with `.dem` files |
| `--output-dir` | required | Output root (frames/, audio/, dataset.json) |
| `--steam-id` | 76561198386265483 | Target player Steam64 ID |
| `--backend` | `wgc` | Capture backend: `wgc` (occluded OK) or `dxcam` (faster, visible only) |
| `--window-name` | `Counter-Strike 2` | CS2 window title |
| `--frame-size` | 640 | Output frame size (square) |
| `--jpeg-quality` | 85 | JPEG quality (1-100) |
| `--no-audio` | false | Disable audio recording |
| `--skip-rounds` | 3 | Skip first N rounds (warmup/knife) |
| `--tick-stride` | 1 | Save every Nth tick (1 = every tick = 64fps) |
| `--saver-threads` | 8 | Parallel JPEG compression threads |
| `--resume` | false | Skip demos that already have frames |
| `--virtual-display` | disabled | Move CS2 to virtual monitor (auto-detect) |
| `--virtual-display N` | — | Move CS2 to monitor index N |
| `--list-devices` | — | Print audio devices and exit |
| `--list-monitors` | — | Print monitors and exit |
| `--validate` | — | Validate a dataset.json and exit |

## Virtual display (recommended)

Running CS2 on a virtual monitor lets you collect data without CS2 interfering with your work. WGC captures the virtual monitor's output just like a real one.

### Setup

1. Install **Virtual Display Driver** (IddSampleDriver):
   https://github.com/itsmikethetech/Virtual-Display-Driver
2. Add a virtual display in Windows Settings -> Display (or via the driver's config)
3. Set CS2 to **Windowed** or **Borderless Windowed** mode

### Usage

```bash
# Auto-detect virtual monitor
python run.py --demo-dir "..." --output-dir "..." --virtual-display

# Use specific monitor index
python run.py --list-monitors          # See available monitors
python run.py --demo-dir "..." --output-dir "..." --virtual-display 1
```

The collector moves CS2 to the virtual monitor in borderless fullscreen, captures via WGC, and restores the window when done.

## Before running

1. **Launch CS2** and get to the main menu
2. **Enable developer console** (Settings -> Game -> Enable Developer Console = Yes)
3. Run the script — it will send console commands to load demos automatically
4. **Press Escape** at any time to stop recording gracefully

Without `--virtual-display`, the collector brings the CS2 window to front briefly to send commands, then pushes it to the background so you can work while it records.

## Architecture

```
run.py                  CLI entry point
  └── collector.py      Main orchestrator (SingleWindowCollector)
        ├── capture.py          WGC/dxcam screen capture (background thread)
        ├── audio.py            WASAPI loopback audio recorder
        ├── demo_control.py     CS2 console commands via pyautogui
        ├── frame_saver.py      ThreadPoolExecutor for async JPEG writes
        ├── tick_clock.py       Wall-clock -> tick mapping (perf_counter)
        ├── virtual_display.py  Move CS2 to virtual monitor (IddSampleDriver)
        └── json_builder.py     dataset.json builder + validator
```

### Per-demo flow

```
Parse .dem (universal_demo_parser)
  → Load demo in CS2 (playdemo, spec_player)
    → For each round:
        Start audio recording
        Start tick clock
        Capture loop (poll at ~200Hz):
          tick_clock.should_capture() → frame_saver.save()
        Stop audio → round_N.wav
    → Flush frame saver
    → Save dataset.json (incremental)
```

### Capture backends

| Backend | Library | Occluded window | Speed | Notes |
|---------|---------|-----------------|-------|-------|
| `wgc` | `windows-capture` | Yes | ~60-120fps | Windows Graphics Capture API |
| `dxcam` | `bettercam` | No | ~120-240fps | DXGI Desktop Duplication |

WGC is the default because it works even when CS2 is behind other windows.

## Resource usage (single window, stride=1)

| Resource | Value |
|----------|-------|
| CS2 RAM | 3-5 GB |
| CS2 VRAM | 1.5-3 GB |
| Python RAM | ~400 MB |
| CPU | ~35-40% total |
| Disk write | ~2-3 MB/s |

### Disk space per demo

| Item | Size |
|------|------|
| 1 frame (640x640 JPEG q85) | ~30-50 KB |
| 1 second (64 frames) | ~2-3 MB |
| 1 round (~2 min) | ~250-380 MB |
| 1 demo (~25 rounds) | ~6-10 GB |
| 100 demos | ~600 GB - 1 TB |

## dataset.json schema

```json
{
  "demos": [
    {
      "demo_path": "D:/dataset/frames/demo_name",
      "audio_path": "D:/dataset/audio/demo_name",
      "rounds": [
        {
          "round_id": 0,
          "start_tick": 4568,
          "end_tick": 8340,
          "states": [
            {
              "tick": 4568,
              "keys": ["W", "MOUSE1"],
              "mouse": [0.5, -0.2],
              "hp": 100,
              "armor": 100,
              "weapon": "AK-47",
              "side": "T",
              "balance": 4200,
              "is_alive": true
            }
          ],
          "events": [
            {"type": "kill", "tick": 5100, "victim_id": 123, "weapon": "ak47", "headshot": true},
            {"type": "death", "tick": 6200, "killer_id": 456, "weapon": "awp"},
            {"type": "damage_dealt", "tick": 5000, "victim_id": 123, "damage": 27, "hitgroup": "4", "weapon": "ak47"},
            {"type": "item_purchase", "tick": 4600, "item": "AK-47"},
            {"type": "weapon_fire", "tick": 5050, "weapon": "ak47"},
            {"type": "bomb_plant", "tick": 7000},
            {"type": "round_end", "tick": 8340, "winner": "T", "reason": "target_bombed"}
          ]
        }
      ]
    }
  ]
}
```

### Event types

| Type | Fields | Description |
|------|--------|-------------|
| `kill` | victim_id, weapon, headshot | Player killed someone |
| `death` | killer_id, weapon | Player died |
| `damage_dealt` | victim_id, damage, hitgroup, weapon | Player dealt damage |
| `damage_taken` | attacker_id, damage, hitgroup, weapon | Player took damage |
| `weapon_fire` | weapon | Player fired weapon |
| `player_blind` | attacker_id, blind_duration | Player was flashed |
| `grenade_thrown` | grenade_type | Player threw a grenade |
| `flashbang_detonate` | — | Player's flashbang detonated |
| `hegrenade_detonate` | — | Player's HE detonated |
| `smokegrenade_detonate` | — | Player's smoke detonated |
| `molotov_detonate` | — | Player's molotov detonated |
| `bomb_begindefuse` | — | Player started defusing |
| `bomb_abortdefuse` | — | Player aborted defuse |
| `item_purchase` | item | Player bought an item |
| `weapon_reload` | — | Player reloaded |
| `bomb_plant` | — | Bomb was planted (global) |
| `bomb_defuse` | — | Bomb was defused (global) |
| `round_end` | winner, reason | Round ended |

## Compatibility

Output is directly compatible with `final_model/DatasetIntent.py`:
- Frames: `tick_{N}.jpg` naming convention
- Radar: cropped from scene frame at `(10, 25, 140, 170)` — no separate capture needed
- Audio: 16kHz mono float32 WAV, `round_{id}.wav`
- JSON: matches `CSRoundDataset` expected schema
