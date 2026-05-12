# Synthetic Data Engine — Deployment Guide

End-to-end instructions for deploying the CS2 multi-bot snapshot reconstruction pipeline on a fresh machine, with notes on scaling across multiple VMs.

---

## 1. System requirements

### Per-instance (1 CS2 + 1 pipeline)

| Component | Min | Recommended |
|---|---|---|
| OS | Windows 10 21H2+ / Windows 11 | Windows 10 IoT Enterprise LTSC 2021 (for VMs) |
| CPU | 4 cores @ 3 GHz | 6+ cores |
| RAM | 12 GB | 16 GB |
| GPU | DX11-capable, 4 GB VRAM | RTX 3050 / GTX 1660 |
| Disk | 80 GB free | 200 GB SSD |
| Network | localhost only | shared SMB for output if scaled |

### Pinned versions (DO NOT auto-update)

- CS2 build **2000807**, manifest `6871635658321698480` (Apr 29 2026)
- Windows binaries depot **2347771**, manifest `8730709599943296195`
- Metamod:Source — version compatible with build 2000807
- CounterStrikeSharp **1.0.367**
- .NET SDK **8.0.x**
- Python **3.13.x**

Auto-updating Steam will break the pinned offsets and signatures.

---

## 2. Install CS2 (pinned build)

### 2.1 Install DepotDownloader

```bash
# Option A: dotnet tool
dotnet tool install --global DepotDownloader

# Option B: standalone release
# Download from https://github.com/SteamRE/DepotDownloader/releases
```

### 2.2 Download pinned CS2

```bash
# Game files (Linux dedicated server depot)
DepotDownloader -app 730 -depot 2347770 -manifest 6871635658321698480 \
    -username <steam_user> -password <steam_pass> \
    -dir D:/CS2_pinned

# Windows binaries (need separately for client)
DepotDownloader -app 730 -depot 2347771 -manifest 8730709599943296195 \
    -username <steam_user> -password <steam_pass> \
    -dir D:/CS2_pinned
```

The `-username/-password` requires a Steam account that owns CS2 (free, but must be authenticated). Use Steam Guard code if prompted.

### 2.3 Disable Steam auto-update for CS2

Steam → Library → CS2 → Properties → Updates → "Only update this game when I launch it" + offline mode if possible. To be safe, **launch CS2 without Steam** (see step 5).

### 2.4 Create launcher script

`D:/CS2_pinned/start_cs2.bat`:

```bat
@echo off
cd /d "D:\CS2_pinned\game\bin\win64"
start cs2.exe -insecure -tools +sv_lan 1 +sv_cheats 1 +map de_mirage
```

The `cd` is critical — CS2 fails to find game files without correct working directory.

---

## 3. Install Metamod:Source + CounterStrikeSharp

### 3.1 Metamod

Download Metamod build matching CS2 2000807:
- https://www.sourcemm.net/downloads.php?branch=master
- Pick a build dated within ±2 weeks of Apr 29 2026

Extract to `D:/CS2_pinned/game/csgo/`:

```
csgo/
├── addons/
│   └── metamod/
│       ├── bin/
│       └── metaplugins.ini
└── gameinfo.gi  (must be patched, see below)
```

Patch `csgo/gameinfo.gi` per Metamod install instructions (add `Game csgo/addons/metamod` line above `Game csgo`).

### 3.2 CounterStrikeSharp

Download CSSharp **1.0.367** Windows release:
- https://github.com/roflmuffin/CounterStrikeSharp/releases/tag/v1.0.367

Extract `addons/counterstrikesharp/` into `D:/CS2_pinned/game/csgo/addons/`.

Verify structure:
```
csgo/addons/counterstrikesharp/
├── api/          (CSSharp managed DLLs)
├── bin/          (native loader)
├── configs/
├── dotnet/       (bundled .NET runtime)
├── gamedata/
│   └── gamedata.json
└── plugins/      (your plugins go here)
```

### 3.3 Verify

Launch CS2 via `start_cs2.bat`. In console:

```
meta list
```

Should show CounterStrikeSharp loaded. If not, check `csgo/addons/metamod/metaplugins.ini` contains `addons/counterstrikesharp/bin/win64/counterstrikesharp`.

---

## 4. Install .NET 8 SDK + Python 3.13

### 4.1 .NET 8 SDK

```bash
# Direct installer
https://dotnet.microsoft.com/download/dotnet/8.0

# Or winget
winget install Microsoft.DotNet.SDK.8
```

Verify: `dotnet --version` → `8.x.x`

### 4.2 Python 3.13

```bash
winget install Python.Python.3.13
```

Verify: `python --version` → `Python 3.13.x`

Install required Python packages:

```bash
pip install pillow demoparser2 pywin32
```

---

## 5. Deploy synthetic_engine

### 5.1 Clone repo

```bash
git clone <your_repo_url> C:/CS2_NN
cd C:/CS2_NN
git checkout audio_branch  # or your active branch
```

### 5.2 Build BotPoseControl plugin

```bash
cd synthetic_engine/plugin/BotPoseControl
dotnet restore
dotnet build -c Release
```

Copy DLL to plugin folder:

```bash
mkdir -p D:/CS2_pinned/game/csgo/addons/counterstrikesharp/plugins/BotPoseControl
cp bin/Release/net8.0/BotPoseControl.dll \
   D:/CS2_pinned/game/csgo/addons/counterstrikesharp/plugins/BotPoseControl/
```

### 5.3 Test plugin loads

Restart CS2. In console:

```
css_plugins list
```

Should show `BotPoseControl` loaded. Check log:
```
type addons/counterstrikesharp/logs/log-BotPoseControl<date>.txt
```

Should contain `TCP listening on 127.0.0.1:27040`.

---

## 6. Calibrate view matrix offset (per-machine)

The view-projection matrix offset in `client.dll` may differ across CS2 builds. Current default: `0x232E9C0`.

To verify on a new install:

```bash
cd C:/CS2_NN/synthetic_engine
python test_bbox_via_matrix.py
```

If projection is wildly off (bboxes miles from screen), use `find_view_matrix_offset.py` to scan for the correct offset. Persist into `--view-matrix-offset` argument.

---

## 7. Prepare snapshot data

### 7.1 Extract from demos

```bash
python extract_snapshots.py \
    --demos "D:/demos/de_mirage/*.dem" \
    --output snapshots.json \
    --map de_mirage \
    --max-demos 50
```

Each snapshot has fields: `viewer`, `kill_target`, `other_players` (bot positions), `planted_bomb`
(`{pos, yaw}` or `null` if no active plant at scenario tick).

Optional: filter to bomb-only scenarios for class balance:
```bash
python -c "import json; d=json.load(open('snapshots.json')); f=[s for s in d if s.get('planted_bomb')]; json.dump(f,open('snapshots_bomb.json','w'),separators=(',',':')); print(f'{len(f)}/{len(d)} kept')"
```

For VM scaling, do extraction **once** on a coordinator machine, then distribute `snapshots.json` to each worker VM.

### 7.2 Optional: shard snapshots for parallel VMs

```python
# split_snapshots.py
import json, sys
n_shards = int(sys.argv[1])
data = json.load(open('snapshots.json'))
for i in range(n_shards):
    shard = data[i::n_shards]
    json.dump(shard, open(f'snapshots_shard{i:02d}.json', 'w'))
```

Each VM works on its shard:
- `snapshots_shard00.json` → VM 0
- `snapshots_shard01.json` → VM 1
- etc.

---

## 8. Run pipeline

### 8.1 Smoke test (1 scenario)

```bash
cd C:/CS2_NN/synthetic_engine
python multibot_pipeline.py \
    --snapshots snapshots.json \
    --output D:/DetectionDataset \
    --map de_mirage \
    --view-matrix-offset 0x232E9C0 \
    --max-scenarios 1
```

Verify `D:/DetectionDataset/frames/scenario_000000.jpg` exists with bot visible.

### 8.2 Visualize annotations

```bash
python visualize_annotations.py --dataset D:/DetectionDataset --max 10
```

Open `D:/DetectionDataset/viz/scenario_*.jpg` and check bboxes match bot silhouettes.

### 8.3 Full run

```bash
python multibot_pipeline.py \
    --snapshots snapshots.json \
    --output D:/DetectionDataset \
    --map de_mirage \
    --view-matrix-offset 0x232E9C0
```

Estimated time: **~35s per scenario** (with pre-break + multi-view enabled).
Each scenario produces **`N_VIEWS_PER_SCENARIO` images** (default 3) named
`scenario_NNNNNN_viewN.jpg` in `frames/`. So 10k scenarios = ~30k labeled images
in ~4 days continuous.

Multi-view config in [multibot_pipeline.py](multibot_pipeline.py) constants:
```python
N_VIEWS_PER_SCENARIO = 3      # original demo aim + 2 random perturbations
VIEW_YAW_RANGE      = 25.0    # ± degrees from original
VIEW_PITCH_RANGE    = 8.0
```

### 8.4 Detection classes (COCO categories)

| ID | Name      | Source                      |
|----|-----------|-----------------------------|
| 0  | `ct`      | CT player body bbox         |
| 1  | `t`       | T player body bbox          |
| 2  | `ct_head` | CT player head bbox         |
| 3  | `t_head`  | T player head bbox          |
| 4  | `bomb`    | Planted C4 entity bbox      |

---

## 9. VM scaling

### 9.1 Architecture

```
┌─────────────────────┐
│  Coordinator (PC)   │  - Holds master snapshots.json
│  - Runs extract     │  - Splits into shards
│  - Aggregates output│  - Mounts shared SMB folder
└──────────┬──────────┘
           │ SMB
   ┌───────┴───────┬───────────┬───────────┐
   ▼               ▼           ▼           ▼
┌──────┐       ┌──────┐    ┌──────┐    ┌──────┐
│ VM 0 │       │ VM 1 │    │ VM 2 │    │ VM N │
│ CS2 +│       │ CS2 +│    │ CS2 +│    │ CS2 +│
│Pipeline      │Pipeline    │Pipeline    │Pipeline
│ shard0       │ shard1     │ shard2     │ shardN
└──────┘       └──────┘    └──────┘    └──────┘
```

### 9.2 Per-VM provisioning checklist

For each VM clone:

1. **OS image** with Windows 10 IoT LTSC 2021 (no auto-updates, no telemetry overhead)
2. **GPU passthrough** (PCIe partitioning if VMware/Hyper-V, or use ASTER Multiseat for shared GPU on bare metal)
3. **Static IP** + hostname (e.g. `cs2-worker-00`)
4. **Mount shared output** as `Z:\` from coordinator via SMB
5. **Install** all software per sections 2-5 above (script this!)
6. **Distribute** the appropriate `snapshots_shardNN.json` to each VM
7. **Start** pipeline:

```bash
cd C:/CS2_NN/synthetic_engine
python multibot_pipeline.py \
    --snapshots snapshots_shard00.json \
    --output Z:/DetectionDataset/shard00 \
    --view-matrix-offset 0x232E9C0
```

Per-VM output goes to its own subfolder to avoid file conflicts.

### 9.3 Coordinator: aggregate annotations

Each VM produces `Z:/DetectionDataset/shardNN/annotations.json` with **per-shard local IDs**. Aggregate script:

```python
# merge_shards.py
import json, glob, os

merged = {'images': [], 'annotations': [], 'categories': None}
img_id = ann_id = 0
for shard_path in sorted(glob.glob('Z:/DetectionDataset/shard*/annotations.json')):
    shard = json.load(open(shard_path))
    if merged['categories'] is None:
        merged['categories'] = shard['categories']

    # Re-key IDs to be globally unique
    img_id_map = {}
    for img in shard['images']:
        img_id += 1
        img_id_map[img['id']] = img_id
        img['id'] = img_id
        # Prefix file_name with shard subdir for unique paths
        shard_dir = os.path.basename(os.path.dirname(shard_path))
        img['file_name'] = f"{shard_dir}/frames/{img['file_name']}"
        merged['images'].append(img)

    for ann in shard['annotations']:
        ann_id += 1
        ann['id'] = ann_id
        ann['image_id'] = img_id_map[ann['image_id']]
        merged['annotations'].append(ann)

json.dump(merged, open('Z:/DetectionDataset/annotations.json', 'w'))
print(f'Merged: {len(merged["images"])} images, {len(merged["annotations"])} annotations')
```

### 9.4 VM provisioning automation

Create a setup PowerShell script that runs on each fresh VM:

```powershell
# provision_vm.ps1
$WORKER_ID = $env:COMPUTERNAME -replace '\D', ''  # extract index from hostname

# 1. Pin Steam (optional: install Steam in offline mode)
# 2. Run DepotDownloader for CS2 + Windows binaries
# 3. Install .NET 8 + Python 3.13 (winget)
# 4. Clone repo
git clone $env:REPO_URL C:\CS2_NN
cd C:\CS2_NN\synthetic_engine\plugin\BotPoseControl
dotnet build -c Release
Copy-Item bin\Release\net8.0\BotPoseControl.dll `
    -Destination D:\CS2_pinned\game\csgo\addons\counterstrikesharp\plugins\BotPoseControl\

# 5. Install Metamod + CSSharp from cached SMB share
robocopy \\coordinator\share\metamod D:\CS2_pinned\game\csgo\addons\metamod /E
robocopy \\coordinator\share\counterstrikesharp D:\CS2_pinned\game\csgo\addons\counterstrikesharp /E

# 6. Pull worker's snapshot shard
Copy-Item "\\coordinator\share\snapshots_shard$WORKER_ID.json" `
    C:\CS2_NN\synthetic_engine\snapshots.json

# 7. Schedule pipeline as Task Scheduler entry on boot
$action = New-ScheduledTaskAction `
    -Execute "python.exe" `
    -Argument "C:\CS2_NN\synthetic_engine\multibot_pipeline.py --snapshots snapshots.json --output Z:\DetectionDataset\shard$WORKER_ID --view-matrix-offset 0x232E9C0"
Register-ScheduledTask -TaskName "CS2DataPipeline" -Trigger (New-ScheduledTaskTrigger -AtStartup) -Action $action
```

### 9.5 Multi-instance on single machine (alternative to VM)

If you have a single beefy box (e.g. RTX 3090 Ti, 256 GB RAM), use **ASTER Multiseat** ($60) to create multiple isolated user sessions sharing the GPU. Each seat runs its own CS2 + pipeline. See `memory/orchestration.md` for details.

Limitation: each seat needs its own TCP port — currently hardcoded to 27040 in [bot_pose_client.py](bot_pose_client.py) and [BotPoseControl.cs](plugin/BotPoseControl/BotPoseControl.cs). To run multiple instances on same machine, parameterize this.

---

## 10. Throughput planning

With multi-view (3 images/scenario) enabled:

| Workers | Scenarios/hour | Images/hour | Time for 10k images | Time for 50k images |
|---|---|---|---|---|
| 1 VM  | ~100  | ~300   | ~33 hours | ~7 days |
| 4 VMs | ~400  | ~1200  | ~8 hours  | ~42 hours |
| 10 VMs| ~1000 | ~3000  | ~3.3 hours | ~17 hours |
| 15 VMs| ~1500 | ~4500  | ~2.2 hours | ~11 hours |

Bottleneck is wall-clock time per scenario (~35s), dominated by visibility check (~12s avg)
and prepare_match/restart_round overhead (~12s combined). Multi-view adds only ~1s
per extra view (no re-shoot, just re-project with new view matrix).

---

## 11. Known issues + mitigations

| Issue | Mitigation |
|---|---|
| Steam auto-updates CS2 → breaks signatures | Launch via `start_cs2.bat` outside Steam; use `-insecure` flag |
| Bot bones empty after spawn (animation not ticking) | `respawn_bots` action force-respawns; pipeline retries automatically |
| Map props get destroyed during visibility check | `pre_break_around_bot` deterministic warm-up shots; replicated post-restart |
| Pass-through textures (visual occlusion without shoot collision) | **Unfixed** — accept noise, plan to clean via active learning |
| View matrix offset shifts across CS2 builds | Recalibrate via `test_bbox_via_matrix.py` after any update |
| TCP port 27040 conflicts on same machine | Parameterize `BOT_POSE_PORT` env var (TODO) for multi-instance |

---

## 12. Quick verification after deployment

```bash
# 1. CS2 launches and loads de_mirage
D:\CS2_pinned\start_cs2.bat

# 2. Plugin shows in console
echo css_plugins list  # → BotPoseControl listed

# 3. Pipeline does 1 scenario
python multibot_pipeline.py --snapshots snapshots.json --output /tmp/test --max-scenarios 1

# 4. Visualize
python visualize_annotations.py --dataset /tmp/test --max 1

# 5. Check timing log shows expected breakdown
#    prepare_match=~5s  place_bots+viewer=~5s  visibility_check=2-12s
#    restart_round+respawn=~6s  pre_break_post_restart=~3s  weapons=~2s
#    screenshot+annotate=~2s (3 views × ~0.7s)
#    Total: ~25-45s per scenario, producing 3 images each (scenario_NNNNNN_view{0,1,2}.jpg)
```

If all 5 pass, you're ready to scale.
