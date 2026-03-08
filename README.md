# CS2 Neural Network Agent

Neural network agent for Counter-Strike 2 that learns to play from demo recordings. The model predicts player actions (mouse movement + key presses) based on visual, audio, and game state information.

## Architecture (~52M parameters)

```
Component                              Params    Status
------------------------------------------------------------
RadarEncoder (EfficientNet-B0)         ~1.0M     trainable
YOLO (YOLOv11l backbone + FPN)         ~25.4M    frozen
YOLO embeds head                       ~1.6M     trainable
StereoAudioEncoder (MN10 / mn10_as)    ~4.88M    trainable
TemporalTransformer (d=384, L=4)       ~19.0M    trainable
FlowActionHead                         ~0.3M     trainable
------------------------------------------------------------
Total: ~52.2M  |  Trainable: ~26.8M
```

### Data Flow

```
[Radar 224x224]     -> RadarEncoder        -> (B, 16, 512)
[Scene 640x640]     -> YOLO frozen         -> P3/P5 features (cached)
                    -> YOLO embeds head    -> (B, 16, 512)
[Scene 640x640]     -> YOLO detections     -> (B, 16, 100)
[Audio 16s stereo]  -> StereoAudioEncoder  -> (B, 16, 512)
[Action history]    -> ModalityCompressor  -> (B, 16, 22)
[Game state]        -> Linear projection   -> (B, 1, 384)
                           |
                    TemporalTransformer (81 tokens)
                           |
                    FlowActionHead -> mouse delta (2)
                    KeysHead       -> key presses (20)
```

### Modalities

| Modality   | Raw Sequence | After Transform | Coverage     |
|------------|-------------|-----------------|--------------|
| Radar      | multi-res   | linspace -> 16  | ~31 sec      |
| Scene      | 64 frames   | cross-attn -> 16| 1-6 sec (T)  |
| Audio      | 32 embeds   | linspace -> 16  | 16 sec       |
| Detection  | 64 frames   | linspace -> 16  | T-synced     |
| Action     | 64 windows  | cross-attn -> 16| 1-6 sec (T)  |
| State      | 100 scalars | proj -> 1 token | current      |

## Project Structure

```
final_model/            # Main training code
  Train.py              # Training loop (step-based eval/ckpt, AMP, cosine LR)
  DatasetIntent.py      # Dataset with T-stride augmentation (T=1..6)
  Yolo.py               # YOLOv11l with frozen backbone + trainable embeds
  AudioEncoder.py       # MN10-based stereo audio encoder
  RadarEncoder.py       # EfficientNet-B0 radar encoder
  TemporalTransformer.py # Unified transformer + FlowActionHead

data_collect_v2/        # Automated data collection from CS2 demos
  cs2_cmd.py            # Direct console input via scan codes
  capture.py            # WGC screen capture
  audio.py              # WASAPI loopback audio recording
  collector.py          # Round recording orchestrator
  run.py                # CLI entry point

inference_pipeline/     # Real-time inference
  inference/engine.py   # Model inference engine
  capture/              # Screen + audio capture
  gsi/gsi_server.py     # Game State Integration server
  buy/buy_executor.py   # Buy round automation
  main.py               # Main entry point

buy_prediction/         # Buy round prediction model
tools/                  # Utilities
  clean_dataset.py      # Remove missing frames from dataset JSON
  warmup_cache.py       # Feature cache warmup
  reparse_and_merge.py  # Demo reparse and merge tool

audio_adaptation/       # Legacy architecture (not used in training)
alive_digit/            # Alive player digit recognition
```

## Training

```bash
cd final_model && python Train.py
```

### Key Parameters

| Parameter       | Value   | Description                         |
|----------------|---------|-------------------------------------|
| BATCH_SIZE     | 4       | Per-GPU batch size                  |
| LR             | 1e-4    | AdamW learning rate                 |
| WARMUP_STEPS   | 1000    | Linear LR warmup                   |
| COSINE_PERIOD  | 100,000 | Cosine decay period                 |
| EVAL_EVERY     | 10,000  | Mid-epoch validation interval       |
| CKPT_EVERY     | 10,000  | Mid-epoch checkpoint interval       |
| T_min / T_max  | 1 / 6   | Temporal stride range (FPS emulation)|

### Loss Functions

| Output       | Loss              | Notes                              |
|-------------|-------------------|------------------------------------|
| Mouse delta | Flow Matching     | Per-tick rate (normalized by T)    |
| Keys (20)   | BCEWithLogitsLoss | Per-class pos_weight, capped at 50 |

### Predicted Actions (20 keys)

`fire`, `second_fire`, `forward`, `back`, `left`, `right`, `jump`, `crouch`, `shift`, `weapon1`, `weapon2`, `weapon3`, `c4`, `reload`, `he`, `molotov`, `smoke`, `flash`, `decoy`, `use`

## Data Collection

Automated pipeline for recording CS2 demo playback:

```bash
cd data_collect_v2 && python run.py --help
```

- Screen capture via Windows Graphics Capture API
- Stereo audio via WASAPI loopback (16kHz)
- Game state via demo parser
- Sandboxie support for multi-instance

## Inference

```bash
cd inference_pipeline && python main.py
```

Requires CS2 running with GSI (Game State Integration) configured.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (training: A100 recommended, inference: RTX 3050+)
- torchaudio, torchvision, opencv-python, ultralytics
