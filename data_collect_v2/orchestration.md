# Orchestration: Multi-Instance CS2 Bot Farm

## Hardware Target
- **Server**: RTX 3090 Ti (24GB VRAM), 48c/96t, 256GB RAM, 2TB SSD
- **Dev machine**: RTX 3050 (4GB VRAM) — single instance only

## Resource Budget Per Instance (800x600 Low, -insecure)

| Resource | Per Instance | Notes |
|----------|-------------|-------|
| VRAM | ~2.0 GB (first), ~1.3 GB (subsequent, shared textures) | VRAM leak: +1-2GB/30min, restart every 3 demos |
| RAM | ~5.0 GB (OS 1.2 + CS2 3.5 + worker 0.3) | |
| CPU | ~2 threads | Demo playback, not full game |
| Disk | ~15 GB/hr per instance | 640x640 JPEG @64fps |

## CS2 Launch Parameters
```
-w 800 -h 600 -novid -low -insecure -nojoy -dx11
+fps_max 64 +r_drawparticles 0 +cl_disablehtmlmotd 1
```

## Recommended OS for VMs/Sessions
**Windows 10 IoT Enterprise LTSC 2021** — minimal bloat, ~1.2GB idle RAM after debloat, full DX11, 10yr support.

---

## Option A: ASTER Multiseat (~$60 license)

### How It Works
- Creates isolated "seats" on one bare-metal machine
- Each seat: own virtual monitor + keyboard + mouse + audio
- All seats share one GPU natively (driver manages VRAM, texture dedup)
- No virtualization overhead

### Architecture
```
Physical Machine (Windows 10 LTSC)
├── Seat 1: ViGEm KB/Mouse → CS2 #1 → WGC Capture + WASAPI Audio
├── Seat 2: ViGEm KB/Mouse → CS2 #2 → WGC Capture + WASAPI Audio
├── ...
└── Seat N: ViGEm KB/Mouse → CS2 #N → WGC Capture + WASAPI Audio
    └── Orchestrator (Python) manages all seats
```

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Multiseat | ASTER v7 |
| Virtual input | ViGEmBus (open source, emulates gamepad/KB/mouse) |
| Virtual monitors | IddSampleDriver (open source) |
| Capture | WGC (per-window, already implemented) |
| Audio | WASAPI loopback (per-session in ASTER) |
| Control | Python (extended collector) |

### Pros
- Proven solution, works on bare metal
- Full GPU sharing with texture deduplication (~30-40% VRAM savings)
- No virtualization overhead
- VAC safe — no CS2 modifications

### Cons
- Paid license ($60)
- Windows 10 only (compatible with LTSC 2021)
- Requires ViGEm for virtual input devices

### Capacity (RTX 3090 Ti)
- Comfortable: ~15 instances (VRAM ≤ 22GB)
- With overcommit: ~18-20 (swap to RAM, FPS drops to 30-40)

---

## Option B: RDP Multi-Session (free, built-in)

### How It Works
- Patch Windows 10 for concurrent RDP sessions (RDP Wrapper)
- Each RDP session = fully isolated Windows session
- Enable GPU in RDP via Group Policy
- Each session runs its own CS2 + agent_worker.py

### Architecture
```
Physical Machine (Windows 10 LTSC)
│
├── Session 0 (Console) — Orchestrator Python (TCP :9000)
│   └── AI Model inference on GPU
│
├── Session 1 (RDP localhost) — User: cs2_bot_01
│   ├── CS2 #1 (GPU DX11)
│   └── agent_worker.py (WGC + WASAPI + SendInput)
│
├── Session 2 (RDP localhost) — User: cs2_bot_02
│   ├── CS2 #2 (GPU DX11)
│   └── agent_worker.py
│
└── Session N ...
```

### Setup Steps
1. Install RDP Wrapper (stascorp/rdpwrap) for concurrent sessions
2. Enable GPU in RDP sessions:
   ```
   # gpedit.msc → Computer Config → Admin Templates → Windows Components
   #   → Remote Desktop Services → Session Host → Remote Session Environment
   #   → "Use hardware graphics adapters for all RDS sessions" = Enabled
   ```
   Or registry:
   ```
   reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services" /v bEnumerateHWBeforeSW /t REG_DWORD /d 1 /f
   ```
3. Create user accounts: `cs2_bot_01` ... `cs2_bot_N`
4. Auto-connect via FreeRDP or mstsc

### Communication
- Orchestrator (Session 0) ↔ Workers (Sessions 1-N) via TCP localhost
- Workers send frames/audio to orchestrator
- Orchestrator runs batch inference, sends actions back

### Critical Check Before Committing
**Must verify**: CS2 renders on GPU in RDP session
1. Install RDP Wrapper
2. Enable GPU policy
3. RDP to localhost as second user
4. Launch CS2 with `-dx11 -w 800 -h 600 -insecure`
5. Check `mat_info` in CS2 console — must show RTX 3090 Ti

### Pros
- Free (RDP Wrapper is open source)
- Full session isolation (input, audio, display)
- Native Windows mechanism, no kernel drivers
- VAC safe

### Cons
- RDP Wrapper may break after Windows Update (LTSC mitigates this)
- GPU in RDP sessions is not guaranteed for all DX11 apps — **must test with CS2**
- CS2 Vulkan might not work in RDP → force `-dx11`
- Need separate Steam account per session, or offline `-insecure` mode
- Slightly more complex setup than ASTER

### Capacity
Same as ASTER — GPU VRAM is the bottleneck, not the session mechanism.

---

## Comparison

| Factor | ASTER | RDP Multi-Session |
|--------|-------|-------------------|
| Cost | $60 | Free |
| GPU support | Guaranteed | Must verify |
| Complexity | Simple setup | More setup steps |
| Stability | Proven | RDP Wrapper can break on updates |
| Input isolation | Via seats | Via sessions |
| Audio isolation | Per-seat | Per-session |
| OS | Windows 10 only | Windows 10/Server |

## Recommendation
1. **Test RDP first** (free) — if CS2 renders on GPU in RDP session, use it
2. **Fallback to ASTER** if RDP GPU doesn't work with CS2
3. For data collection (demo playback), current single-instance approach works fine

## VRAM Optimization Tips
- All instances on bare metal (not VMs) → driver texture dedup (~30-40% savings)
- Launch params: `-w 800 -h 600 -novid -low -dx11 +r_drawparticles 0`
- Restart CS2 every 3 demos (VRAM leak bug)
- Monitor with `nvidia-smi`
- `-vulkan` may save 200-400MB but untested in RDP sessions
