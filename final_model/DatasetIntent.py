"""
CSRoundDataset — Fixed-size sequence outputs for the final architecture.

Key changes from audio_adaptation/src/DatasetIntent.py:
  - All sequences always return exactly SEQ_LEN=16 frames (linspace internally)
  - radar_window: 32 frames @1Hz (32 sec) → always 16 after linspace
  - scene_window: 64 frames               → always 16 after linspace
  - actions_window: 64 windows            → always 16 after linspace
  - Returns frame paths for backbone caching in Train.py:
      scene_frame_paths: list[str] len=16
      radar_frame_paths: list[str] len=16
  - Audio: 16 sec window (256,000 samples @ 16kHz) for AudioEncoder
  - No ExactMatchBucketSampler needed — all sequences are fixed size 16
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import random
from typing import Optional, Tuple, Dict, Any, List

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("[WARNING] torchaudio not installed. Audio features disabled.")


SEQ_LEN = 16          # Radar, detection, audio: 16 tokens
SCENE_SEQ_LEN = 64   # Scene: 64 tokens (~4 sec @ ~16 Hz) → compressed 64→16 in Transformer
ACTION_SEQ_LEN = 64  # Action history: 64 tokens (~4 sec) → compressed 64→16 in Transformer


class CSRoundDataset(Dataset):
    """
    CS2 Round Dataset — fixed-size sequences for the final 50M architecture.

    All outputs have fixed size SEQ_LEN=16, enabling simple DataLoader batching
    without ExactMatchBucketSampler.

    Output keys:
        game_id, round_id, demo_path, tick
        scene_frame_paths: list[str] length 64 (paths for Train.py to load)
        radar_seq:       (16, 224, 224, 3) float32
        radar_frame_paths: list[str] length 16
        audio_waveform:  (2, 256000) float32 stereo  or zeros if not available
        actions_mouse:   (64, 2) float32
        actions_keys:    (64, 20) float32
        state_vec:       (100,) float32
        intent:          (20,) float32
        target_mouse:    (2,) float32  (per-tick rate: delta / (actual_ticks * MOUSE_SCALE))
        T:               int  (temporal stride in ticks)
    """

    AUDIO_SAMPLE_RATE = 16000
    AUDIO_WINDOW_SEC = 16.0                         # 16 sec audio window
    AUDIO_WINDOW_SAMPLES = int(AUDIO_WINDOW_SEC * AUDIO_SAMPLE_RATE)  # 256,000
    BASE_TICK_RATE = 64

    def __init__(
        self,
        dataset_json: str,
        T_min: int = 1,
        T_max: int = 6,
        radar_window: int = 32,    # 32 raw @1Hz → linspace to 16
        scene_window: int = 64,    # 64 raw frames → linspace to 16
        actions_window: int = 64,  # 64 raw windows → linspace to 16
        transform_scene=None,
        transform_radar=None,
        radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170),
        use_audio: bool = True,
        audio_dir: Optional[str] = None,
        audio_speedup_factor: float = 1.0,
        img_size: int = 640,        # scene image size for resize
    ):
        self.T_min = T_min
        self.T_max = T_max
        self.radar_window = radar_window
        self.scene_window = scene_window
        self.actions_window = actions_window
        self.transform_scene = transform_scene
        self.transform_radar = transform_radar
        self.radar_crop_box = radar_crop_box
        self.img_size = img_size

        self.use_audio = use_audio and TORCHAUDIO_AVAILABLE
        self.audio_dir = audio_dir
        self.audio_speedup_factor = audio_speedup_factor
        self.TICK_RATE = self.BASE_TICK_RATE * audio_speedup_factor

        self._audio_cache: Dict[str, torch.Tensor] = {}
        self._audio_cache_max_size = 10

        self.KEYS = [
            "MOUSE_LEFT", "SPACE", "CTRL", "W", "S", "E", "A", "D",
            "MOUSE_RIGHT", "R", "MOLOTOV", "TAB", "F",
            "WEAPON1", "WEAPON2", "WEAPON3",
            "HE", "FLASH", "SMOKE", "DECOY", "C4", "SHIFT"
        ]
        self.WEAPONS = [
            'p2000', 'usp-s', 'p250', 'five-seven', 'glock-18', 'tec-9',
            'cz75-auto', 'dual berettas', 'desert eagle', 'm249',
            'r8 revolver', 'mp9', 'mac-10', 'pp-bizon', 'mp7',
            'ump-45', 'p90', 'mp5-sd', 'famas', 'galil ar', 'sawed-off',
            'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553', 'ssg 08',
            'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
            'negev', 'knife', 'high explosive grenade', 'flashbang',
            'smoke grenade', 'decoy grenade', 'molotov',
            'incendiary grenade', 'c4 explosive', 'None'
        ]
        self.SIDES = ["CT", "T"]
        self.MOUSE_SCALE = 25.0
        self.allowed_T = list(range(T_min, T_max + 1))

        self.metadata: List[Dict] = []
        self.samples: List[Dict] = []
        self._load_metadata(dataset_json)
        self._build_samples_index()

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def _load_metadata(self, dataset_json: str):
        with open(dataset_json, 'r', encoding='utf-8') as f:
            full_meta = json.load(f)

        for game in full_meta['demos']:
            demo_path = game['demo_path']
            game_id = os.path.basename(demo_path)

            audio_path = game.get('audio_path', None)
            if audio_path is None and self.audio_dir:
                audio_path = os.path.join(self.audio_dir, game_id)

            for rnd in game['rounds']:
                round_audio_path = None
                if audio_path:
                    f = os.path.join(audio_path, f"round_{rnd['round_id']}.wav")
                    if os.path.exists(f):
                        round_audio_path = f

                start_tick = rnd.get('start_tick',
                                     rnd['states'][0]['tick'] if rnd['states'] else 0)
                end_tick = rnd.get('end_tick',
                                   rnd['states'][-1]['tick'] if rnd['states'] else start_tick)
                drift_ticks = rnd.get('drift_info', {}).get('drift_ticks', 0)

                self.metadata.append({
                    'game_id': game_id,
                    'round_id': rnd['round_id'],
                    'demo_path': demo_path,
                    'audio_path': round_audio_path,
                    'start_tick': start_tick,
                    'end_tick': end_tick,
                    'drift_ticks': drift_ticks,
                    'states': rnd['states'],
                })

    MIN_FRAMES_PER_ROUND = 2000

    def _count_round_frames(self, item: Dict) -> int:
        """Count how many frame files actually exist for this round."""
        states = item['states']
        if not states:
            return 0
        # Sample every 4th tick to avoid slow full scan (approximate count only)
        count = 0
        for idx in range(0, len(states), 4):
            tick = states[idx]['tick']
            if os.path.exists(self._frame_path(item['demo_path'], tick)):
                count += 1
        return count * 4  # approximate total

    def _build_samples_index(self):
        skipped_rounds = 0
        for item in self.metadata:
            frame_count = self._count_round_frames(item)
            if frame_count < self.MIN_FRAMES_PER_ROUND:
                skipped_rounds += 1
                continue
            states = item['states']
            for i in range(len(states)):
                self.samples.append({
                        'game_id': item['game_id'],
                        'round_id': item['round_id'],
                        'demo_path': item['demo_path'],
                        'audio_path': item['audio_path'],
                        'start_tick': item['start_tick'],
                        'end_tick': item['end_tick'],
                        'drift_ticks': item['drift_ticks'],
                        'states': states,
                        'tick_idx': i,
                    })

        audio_count = sum(1 for s in self.samples if s['audio_path'] is not None)
        print(f"[INFO] {len(self.samples)} samples ({audio_count} with audio), "
              f"{skipped_rounds} rounds skipped (< {self.MIN_FRAMES_PER_ROUND} frames)")

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------ #
    # Frame helpers                                                        #
    # ------------------------------------------------------------------ #

    def _apply_drift(self, tick: int, start_tick: int, end_tick: int, drift_ticks: int) -> int:
        """Linear drift correction: 0 at round start, drift_ticks at round end."""
        if drift_ticks == 0:
            return tick
        progress = (tick - start_tick) / max(1, end_tick - start_tick)
        return tick + round(progress * drift_ticks)

    def _frame_path(self, demo_path: str, tick: int) -> str:
        return os.path.join(demo_path, f"tick_{tick}.jpg")

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image as float32 RGB, or None if file is missing/corrupt."""
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _crop_radar(self, image: np.ndarray) -> np.ndarray:
        l, t, r, b = self.radar_crop_box
        img = image[t:b, l:r, :]
        return cv2.resize(img, (224, 224))

    def _linspace_indices(self, n: int, target: int = SEQ_LEN) -> List[int]:
        """Return `target` equally spaced indices from [0, n-1]."""
        if n <= 0:
            return [0] * target
        return list(np.linspace(0, n - 1, target, dtype=int))

    # ------------------------------------------------------------------ #
    # Encoding helpers                                                     #
    # ------------------------------------------------------------------ #

    def _encode_weapon(self, weapon: str) -> np.ndarray:
        vec = np.zeros(len(self.WEAPONS), dtype=np.float32)
        w = weapon.lower()
        if w in self.WEAPONS:
            vec[self.WEAPONS.index(w)] = 1.0
        elif any(k in w for k in ('knife', 'bayonet', 'karambit', 'daggers')):
            vec[self.WEAPONS.index('knife')] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def _encode_weapon_list(self, weapon_list) -> np.ndarray:
        vec = np.zeros(len(self.WEAPONS), dtype=np.float32)
        for weapon in weapon_list:
            w = weapon.lower()
            if w in self.WEAPONS:
                vec[self.WEAPONS.index(w)] = 1.0
            elif any(k in w for k in ('knife', 'bayonet', 'karambit', 'daggers')):
                vec[self.WEAPONS.index('knife')] = 1.0
            else:
                vec[-1] = 1.0
        return vec

    def _encode_side(self, side: str) -> np.ndarray:
        vec = np.zeros(len(self.SIDES), dtype=np.float32)
        if side in self.SIDES:
            vec[self.SIDES.index(side)] = 1.0
        return vec

    def _safe_value(self, value, default=0.0) -> float:
        return float(value) if value is not None else default

    def _compute_T(self, idx: int) -> int:
        return random.Random(idx).choice(self.allowed_T)

    def _build_intent(self, keys_window: set) -> Dict[str, float]:
        intent = {}
        intent['fire'] = 1.0 if 'MOUSE_LEFT' in keys_window else 0.0
        intent['second_fire'] = 1.0 if 'MOUSE_RIGHT' in keys_window else 0.0
        intent['forward'] = 1.0 if 'W' in keys_window else 0.0
        intent['back'] = 1.0 if 'S' in keys_window else 0.0
        intent['left'] = 1.0 if ('A' in keys_window and 'D' not in keys_window) else 0.0
        intent['right'] = 1.0 if ('D' in keys_window and 'A' not in keys_window) else 0.0
        intent['jump'] = 1.0 if 'SPACE' in keys_window else 0.0
        intent['crouch'] = 1.0 if 'CTRL' in keys_window else 0.0
        intent['shift'] = 1.0 if 'SHIFT' in keys_window else 0.0
        for key in ['WEAPON1', 'WEAPON2', 'WEAPON3', 'C4', 'R']:
            intent[key.lower()] = 1.0 if key in keys_window else 0.0
        for key in ['HE', 'MOLOTOV', 'SMOKE', 'FLASH', 'DECOY']:
            intent[key.lower()] = 1.0 if key in keys_window else 0.0
        intent['use'] = 1.0 if 'E' in keys_window else 0.0
        return intent

    # ------------------------------------------------------------------ #
    # Audio                                                                #
    # ------------------------------------------------------------------ #

    def _load_audio_cached(self, audio_path: str) -> Optional[torch.Tensor]:
        if audio_path is None:
            return None
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Keep stereo (2, T) — spatial audio cues (ILD, HRTF) require both channels
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)   # mono fallback → pseudo-stereo
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]          # keep first 2 channels only
            # waveform: (2, T)
            if sr != self.AUDIO_SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.AUDIO_SAMPLE_RATE)(waveform)
            # waveform: (2, T_resampled)
            if len(self._audio_cache) >= self._audio_cache_max_size:
                del self._audio_cache[next(iter(self._audio_cache))]
            self._audio_cache[audio_path] = waveform
            return waveform
        except Exception as e:
            print(f"[WARNING] Failed to load audio {audio_path}: {e}")
            return None

    def _get_audio_window(self, audio_path: str, current_tick: int,
                          round_start_tick: int) -> torch.Tensor:
        # Stereo zeros: (2, AUDIO_WINDOW_SAMPLES)
        zeros = torch.zeros(2, self.AUDIO_WINDOW_SAMPLES, dtype=torch.float32)
        if not self.use_audio:
            return zeros

        waveform = self._load_audio_cached(audio_path)
        if waveform is None:
            return zeros

        # waveform: (2, T_total)
        ticks_since_start = current_tick - round_start_tick
        time_sec = ticks_since_start / self.TICK_RATE
        current_sample = int(time_sec * self.AUDIO_SAMPLE_RATE)

        start_sample = max(0, current_sample - self.AUDIO_WINDOW_SAMPLES)
        end_sample = min(current_sample, waveform.shape[1])

        audio_window = waveform[:, start_sample:end_sample]   # (2, window)

        if audio_window.shape[1] < self.AUDIO_WINDOW_SAMPLES:
            pad = self.AUDIO_WINDOW_SAMPLES - audio_window.shape[1]
            audio_window = torch.cat(
                [torch.zeros(2, pad, dtype=torch.float32), audio_window], dim=1
            )

        return audio_window[:, :self.AUDIO_WINDOW_SAMPLES]   # (2, AUDIO_WINDOW_SAMPLES)

    # ------------------------------------------------------------------ #
    # __getitem__                                                          #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        states = sample['states']
        demo_path = sample['demo_path']
        i = sample['tick_idx']
        start_tick  = sample['start_tick']
        end_tick    = sample['end_tick']
        drift_ticks = sample['drift_ticks']

        T = self._compute_T(idx)
        t_start = max(0, i - T + 1)

        def frame_path(tick: int) -> str:
            t = self._apply_drift(tick, start_tick, end_tick, drift_ticks)
            return self._frame_path(demo_path, t)

        # ---- SCENE: up to scene_window raw frames → 64 tokens ----
        raw_scene_indices = list(range(max(0, i - (self.scene_window - 1) * T), i + 1, T))
        chosen_scene = self._linspace_indices(len(raw_scene_indices), target=SCENE_SEQ_LEN)

        scene_paths = []
        _last_scene_path: Optional[str] = None
        _scene_fallback = 0
        _scene_black = 0
        for k in chosen_scene:
            j = raw_scene_indices[k]
            tick = states[j]['tick']
            p = frame_path(tick)
            if os.path.exists(p):
                _last_scene_path = p
                scene_paths.append(p)
            elif _last_scene_path is not None:
                scene_paths.append(_last_scene_path)  # repeat last valid
                _scene_fallback += 1
            else:
                scene_paths.append(p)  # black frame (no valid frame seen yet)
                _scene_black += 1
        if _scene_fallback or _scene_black:
            print(f"[WARN] scene missing frames — "
                  f"fallback={_scene_fallback}, black={_scene_black} "
                  f"| {sample['game_id']} round={sample['round_id']} tick={states[i]['tick']}")

        # ---- RADAR: up to radar_window raw frames @1Hz ----
        raw_radar_indices = list(range(max(0, i - self.radar_window * 64 + 1), i + 1, 64))
        if not raw_radar_indices or raw_radar_indices[-1] != i:
            raw_radar_indices.append(i)
        chosen_radar = self._linspace_indices(len(raw_radar_indices))

        radar_frames = []
        radar_paths = []
        _last_radar: Optional[np.ndarray] = None
        for k in chosen_radar:
            j = raw_radar_indices[k]
            tick = states[j]['tick']
            path = frame_path(tick)
            radar_paths.append(path)
            img = self._load_image(path)
            if img is None:
                radar = _last_radar if _last_radar is not None else \
                    np.zeros((224, 224, 3), dtype=np.float32)
            else:
                radar = self._crop_radar(img)
                _last_radar = radar
            if self.transform_radar:
                radar = self.transform_radar(radar)
            radar_frames.append(radar)

        radar_seq = torch.tensor(np.stack(radar_frames), dtype=torch.float32)

        # ---- AUDIO ----
        current_tick = states[i]['tick']
        audio_waveform = self._get_audio_window(
            sample['audio_path'], current_tick, sample['start_tick']
        )

        # ---- STATE (latest observation) ----
        st = states[i]
        state_vec = np.concatenate([
            np.array([
                st['hp'] / 100.0,
                st['armor'] / 100.0,
                float(st['helmet']),
                self._safe_value(st.get('ammo'), 0.0) / 100.0,
                st['ct_alive'] / 5.0,
                st['t_alive'] / 5.0,
                self._safe_value(st.get('round_time_left'), 0.0) / 115.0,
                float(self._safe_value(st.get('bomb_planted'), 0.0)),
                float(self._safe_value(st.get('freeze_time'), 0.0)),
                float(self._safe_value(st.get('defuser'), 0.0)),
                self._safe_value((st.get('score') or [0, 0])[0], 0.0) / 16.0,
                self._safe_value((st.get('score') or [0, 0])[1], 0.0) / 16.0,
            ]),
            self._encode_side(st['side']),
            self._encode_weapon(st['weapon']),
            self._encode_weapon_list(st['weapon_list']),
        ]).astype(np.float32)
        state_vec = torch.from_numpy(state_vec)

        # ---- ACTION HISTORY: up to actions_window raw windows ----
        raw_intent_keys = []
        raw_intent_mouse = []
        for k in range(self.actions_window):
            end = i - (k + 1) * T
            start = max(0, end - T + 1)
            if end < 0:
                break
            keys_window = set()
            mouse_start = mouse_end = None
            for j in range(start, end + 1):
                s = states[j]
                if mouse_start is None:
                    mouse_start = np.array(s['mouse'], dtype=np.float32)
                mouse_end = np.array(s['mouse'], dtype=np.float32)
                keys_window.update(s['keys'])

            intent_d = self._build_intent(keys_window)
            raw_intent_keys.append(torch.tensor(list(intent_d.values()), dtype=torch.float32))
            window_size = end - start + 1
            mouse_delta = (mouse_end - mouse_start) / (max(1, window_size) * self.MOUSE_SCALE)
            raw_intent_mouse.append(torch.tensor(mouse_delta, dtype=torch.float32))

        # Zero-pad at start if fewer than actions_window
        while len(raw_intent_keys) < self.actions_window:
            n_k = raw_intent_keys[0].shape[0] if raw_intent_keys else 20
            raw_intent_keys.append(torch.zeros(n_k, dtype=torch.float32))
            raw_intent_mouse.append(torch.zeros(2, dtype=torch.float32))

        raw_intent_keys = torch.stack(raw_intent_keys[::-1])    # (actions_window, 20)
        raw_intent_mouse = torch.stack(raw_intent_mouse[::-1])  # (actions_window, 2)

        # Linspace actions_window → ACTION_SEQ_LEN
        act_idx = self._linspace_indices(self.actions_window, target=ACTION_SEQ_LEN)
        actions_keys = raw_intent_keys[act_idx]    # (ACTION_SEQ_LEN, 20)
        actions_mouse = raw_intent_mouse[act_idx]  # (ACTION_SEQ_LEN, 2)

        # ---- TARGET INTENT ----
        target_keys = set()
        for j in range(t_start, i + 1):
            target_keys.update(states[j]['keys'])
        intent_d = self._build_intent(target_keys)
        intent_vec = torch.tensor(list(intent_d.values()), dtype=torch.float32)

        # ---- TARGET MOUSE ----
        yaw_now = states[i]['mouse'][0]
        pitch_now = states[i]['mouse'][1]
        yaw_prev = states[t_start]['mouse'][0]
        pitch_prev = states[t_start]['mouse'][1]
        actual_ticks = max(1, i - t_start + 1)  # actual window (< T at round start)
        target_mouse = torch.tensor(
            [(yaw_now - yaw_prev) / (actual_ticks * self.MOUSE_SCALE),
             (pitch_now - pitch_prev) / (actual_ticks * self.MOUSE_SCALE)],
            dtype=torch.float32
        )

        return {
            'game_id': sample['game_id'],
            'round_id': sample['round_id'],
            'demo_path': sample['demo_path'],
            'tick': states[i]['tick'],
            'scene_frame_paths': scene_paths,         # list[str] len=64
            'radar_seq': radar_seq,                   # (16, 224, 224, 3)
            'radar_frame_paths': radar_paths,          # list[str] len=16
            'audio_waveform': audio_waveform,          # (2, 256000)
            'actions_mouse': actions_mouse,             # (64, 2)
            'actions_keys': actions_keys,               # (64, 20)
            'state_vec': state_vec,                    # (100,)
            'intent': intent_vec,                      # (20,)
            'target_mouse': target_mouse,              # (2,)
            'T': T,
        }
