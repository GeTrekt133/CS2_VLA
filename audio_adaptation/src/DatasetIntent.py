"""
CSRoundDataset with Audio Support.

Updated dataset to include audio data synchronized with game ticks.
Audio window: 30 seconds ending at current tick.

New output: audio_waveform (480000,) - 30 sec @ 16kHz mono
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import random
from typing import Optional, Tuple, Dict, Any

# Optional audio support
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("[WARNING] torchaudio not installed. Audio features disabled.")


class CSRoundDataset(Dataset):
    """
    CS2 Round Dataset with Audio Support.

    Loads:
    - Scene frames (from video)
    - Radar frames (cropped from scene)
    - Game state
    - Action history
    - Audio waveform (NEW)

    Audio specs:
    - Window: 30 seconds
    - Sample rate: 16000 Hz
    - Output: (480000,) mono waveform
    """

    # Audio constants (base values, will be adjusted in __init__ based on speedup)
    AUDIO_SAMPLE_RATE = 16000
    BASE_TICK_RATE = 64  # Standard CS2 tickrate
    BASE_AUDIO_WINDOW_SEC = 30.0

    def __init__(
        self,
        dataset_json: str,
        T_min: int = 3,
        T_max: int = 12,
        radar_window: int = 128,
        scene_window: int = 16,
        actions_window: int = 16,
        transform_scene=None,
        transform_radar=None,
        radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170),
        sampler: bool = False,
        # Audio settings (NEW)
        use_audio: bool = True,
        audio_dir: Optional[str] = None,  # If None, looks for audio in demo_path
        audio_speedup_factor: float = 1.0,  # No speedup (recording at normal speed)
    ):
        self.dataset_json = dataset_json
        self.T_min = T_min
        self.T_max = T_max
        self.radar_window = radar_window
        self.scene_window = scene_window
        self.actions_window = actions_window
        self.sampling = sampler

        self.transform_scene = transform_scene
        self.transform_radar = transform_radar
        self.radar_crop_box = radar_crop_box

        # Audio settings
        self.use_audio = use_audio and TORCHAUDIO_AVAILABLE
        self.audio_dir = audio_dir
        self.audio_speedup_factor = audio_speedup_factor

        # Calculate adjusted constants based on speedup
        self.AUDIO_WINDOW_SEC = self.BASE_AUDIO_WINDOW_SEC / audio_speedup_factor  # 30.0 for 1x (no speedup)
        self.AUDIO_WINDOW_SAMPLES = int(self.AUDIO_WINDOW_SEC * self.AUDIO_SAMPLE_RATE)  # 480000
        self.TICK_RATE = self.BASE_TICK_RATE * audio_speedup_factor  # 64 for 1x (no speedup)

        # Audio cache to avoid reloading
        self._audio_cache = {}
        self._audio_cache_max_size = 10  # Keep last N audio files in memory

        self.metadata = []
        self.samples = []
        self._load_metadata()
        self._build_samples_index()

        # ====== CONSTANTS ======
        self.KEYS = [
            "MOUSE_LEFT", "SPACE", "CTRL", "W", "S", "E", "A", "D",
            "MOUSE_RIGHT", "R", "MOLOTOV", "TAB", "F",
            "WEAPON1", "WEAPON2", "WEAPON3",
            "HE", "FLASH", "SMOKE", "DECOY", "C4", "SHIFT"
        ]

        self.WEAPONS = [
            'p2000', 'p250', 'five-seven', 'glock-18', 'tec-9',
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

        self.allowed_T = [x for x in range(self.T_min, self.T_max + 1) if x % 4 == 0]

        # Mouse normalization scale (approximate std of yaw/pitch deltas)
        # При инференсе нужно умножить предсказание обратно на MOUSE_SCALE
        self.MOUSE_SCALE = 25.0

    # =====================================================
    def _load_metadata(self):
        with open(self.dataset_json, "r", encoding="utf-8") as f:
            full_meta = json.load(f)

        for game in full_meta["demos"]:
            demo_path = game["demo_path"]
            game_id = os.path.basename(demo_path)

            # Check for audio directory
            audio_path = game.get("audio_path", None)
            if audio_path is None and self.audio_dir:
                audio_path = os.path.join(self.audio_dir, game_id)

            for rnd in game["rounds"]:
                # Audio file path for this round
                round_audio_path = None
                if audio_path:
                    round_audio_file = os.path.join(audio_path, f"round_{rnd['round_id']}.wav")
                    if os.path.exists(round_audio_file):
                        round_audio_path = round_audio_file

                self.metadata.append({
                    "game_id": game_id,
                    "round_id": rnd["round_id"],
                    "demo_path": demo_path,
                    "audio_path": round_audio_path,
                    "start_tick": rnd.get("start_tick", rnd["states"][0]["tick"] if rnd["states"] else 0),
                    "states": rnd["states"]
                })

    def _build_samples_index(self):
        for item in self.metadata:
            states = item["states"]
            for i in range(len(states)):
                if i % 4 == 0:
                    self.samples.append({
                        "game_id": item["game_id"],
                        "round_id": item["round_id"],
                        "demo_path": item["demo_path"],
                        "audio_path": item["audio_path"],
                        "start_tick": item["start_tick"],
                        "states": states,
                        "tick_idx": i
                    })

        audio_count = sum(1 for s in self.samples if s["audio_path"] is not None)
        print(f"[INFO] Prepared {len(self.samples)} intent samples ({audio_count} with audio)")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _crop_radar(self, image: np.ndarray) -> np.ndarray:
        l, t, r, b = self.radar_crop_box
        img = image[t:b, l:r, :]
        img = cv2.resize(img, (224, 224))
        return img

    def _encode_keys(self, keys) -> np.ndarray:
        vec = np.zeros(len(self.KEYS), dtype=np.float32)
        for k in keys:
            if k in self.KEYS:
                vec[self.KEYS.index(k)] = 1.0
        return vec

    def _encode_weapon(self, weapon: str) -> np.ndarray:
        vec = np.zeros(len(self.WEAPONS), dtype=np.float32)
        w = weapon.lower()
        if w in self.WEAPONS:
            vec[self.WEAPONS.index(w)] = 1.0
        elif 'knife' in w or 'bayonet' in w or 'karambit' in w or 'daggers' in w:
            vec[self.WEAPONS.index("knife")] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def _encode_weapon_list(self, weapon_list) -> np.ndarray:
        vec = np.zeros(len(self.WEAPONS), dtype=np.float32)
        for weapon in weapon_list:
            w = weapon.lower()
            if w in self.WEAPONS:
                vec[self.WEAPONS.index(w)] = 1.0
            elif 'knife' in w or 'bayonet' in w or 'karambit' in w or 'daggers' in w:
                vec[self.WEAPONS.index('knife')] = 1.0
            else:
                vec[-1] = 1.0
        return vec

    def _encode_side(self, side: str) -> np.ndarray:
        vec = np.zeros(len(self.SIDES), dtype=np.float32)
        if side in self.SIDES:
            vec[self.SIDES.index(side)] = 1.0
        return vec

    def _safe_value(self, value, default: float = 0.0) -> float:
        return float(value) if value is not None else default

    def _compute_T(self, idx: int) -> int:
        rng = random.Random(idx)
        return rng.choice(self.allowed_T)

    # =====================================================
    # AUDIO LOADING (NEW)
    # =====================================================
    def _load_audio_cached(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load audio with caching."""
        if audio_path is None:
            return None

        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)  # (num_samples,)

            # Cache management
            if len(self._audio_cache) >= self._audio_cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]

            self._audio_cache[audio_path] = waveform
            return waveform

        except Exception as e:
            print(f"[WARNING] Failed to load audio {audio_path}: {e}")
            return None

    def _get_audio_window(
        self,
        audio_path: str,
        current_tick: int,
        round_start_tick: int
    ) -> torch.Tensor:
        """
        Extract audio window ending at current tick.
        Window duration adjusted for audio speedup factor.
        Output is padded to fixed size (480000) for AudioEncoder.

        For 4x speedup:
        - Extracts 7.5 seconds of accelerated audio (120000 samples)
        - Represents 30 seconds of game time
        - Pads to (480000,) for AudioEncoder compatibility

        Args:
            audio_path: Path to round audio file
            current_tick: Current game tick
            round_start_tick: First tick of the round

        Returns:
            audio_window: (480000,) tensor or zeros if not available
        """
        FIXED_ENCODER_INPUT = 480000

        if not self.use_audio:
            return torch.zeros(FIXED_ENCODER_INPUT, dtype=torch.float32)

        waveform = self._load_audio_cached(audio_path)

        if waveform is None:
            return torch.zeros(FIXED_ENCODER_INPUT, dtype=torch.float32)

        # Calculate audio position
        # Ticks since round start
        ticks_since_start = current_tick - round_start_tick

        # Convert ticks to seconds: ticks / 64 = seconds
        time_sec = ticks_since_start / self.TICK_RATE

        # Audio sample position for current tick
        current_sample = int(time_sec * self.AUDIO_SAMPLE_RATE)

        # Window: [current_sample - window_samples, current_sample]
        start_sample = max(0, current_sample - self.AUDIO_WINDOW_SAMPLES)
        end_sample = current_sample

        # Handle edge cases
        total_samples = waveform.shape[0]

        if end_sample > total_samples:
            end_sample = total_samples
            start_sample = max(0, end_sample - self.AUDIO_WINDOW_SAMPLES)

        # Extract window
        audio_window = waveform[start_sample:end_sample]

        # Pad if needed (at start of round)
        if audio_window.shape[0] < self.AUDIO_WINDOW_SAMPLES:
            padding_needed = self.AUDIO_WINDOW_SAMPLES - audio_window.shape[0]
            audio_window = torch.cat([
                torch.zeros(padding_needed, dtype=torch.float32),
                audio_window
            ])

        # Pad to fixed size for AudioEncoder (480000 samples)
        # This ensures compatibility even with speedup_factor > 1
        FIXED_ENCODER_INPUT = 480000
        if audio_window.shape[0] < FIXED_ENCODER_INPUT:
            padding_needed = FIXED_ENCODER_INPUT - audio_window.shape[0]
            audio_window = torch.nn.functional.pad(audio_window, (0, padding_needed))
        elif audio_window.shape[0] > FIXED_ENCODER_INPUT:
            # Should not happen with speedup_factor >= 1, but handle it
            audio_window = audio_window[:FIXED_ENCODER_INPUT]

        return audio_window  # Always (480000,)

    # =====================================================
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.sampling is True:
            i = self.samples[idx]["tick_idx"]
            T = self._compute_T(idx)

            scene_indices = list(range(max(0, i - (self.scene_window - 1) * T), i + 1, T))
            radar_indices = list(range(max(0, i - self.radar_window * 64 + 1), i + 1, 64))

            if radar_indices[-1] != i:
                radar_indices.append(i)
            ln_radar = len(radar_indices)
            ln_scene = len(scene_indices)

            return ln_scene, ln_radar

        sample = self.samples[idx]
        states = sample["states"]
        demo_path = sample["demo_path"]
        i = sample["tick_idx"]

        # =====================================================
        # 1. INTENT WINDOW (random T)
        # =====================================================
        T = self._compute_T(idx)
        t_start = max(0, i - T + 1)

        # =====================================================
        # 2. SCENE (step = T)
        # =====================================================
        scene_indices = list(range(max(0, i - (self.scene_window - 1) * T), i + 1, T))

        scene_frames = []
        for j in scene_indices:
            tick = states[j]["tick"]
            frame_path = os.path.join(demo_path, f"tick_{tick}.jpg")
            if os.path.exists(frame_path):
                img = self._load_image(frame_path)
                if self.transform_scene:
                    img = self.transform_scene(img)
                scene_frames.append(img)
            else:
                print("Broken img - ", frame_path)
        if len(scene_frames) == 0:
            scene_frames.append(np.zeros((640, 640, 3), dtype=np.float32))

        scene_seq = torch.tensor(np.stack(scene_frames), dtype=torch.float32)

        # =====================================================
        # 3. RADAR (fixed window)
        # =====================================================
        radar_indices = list(range(max(0, i - self.radar_window * 64 + 1), i + 1, 64))
        if radar_indices[-1] != i:
            radar_indices.append(i)
        radar_frames = []
        for j in radar_indices:
            tick = states[j]["tick"]
            frame_path = os.path.join(demo_path, f"tick_{tick}.jpg")
            if os.path.exists(frame_path):
                img = self._load_image(frame_path)
                radar = self._crop_radar(img)
                if self.transform_radar:
                    radar = self.transform_radar(radar)
                radar_frames.append(radar)

        if len(radar_frames) == 0:
            radar_frames.append(np.zeros((224, 224, 3), dtype=np.float32))

        radar_seq = torch.tensor(np.stack(radar_frames), dtype=torch.float32)

        # =====================================================
        # 4. AUDIO (NEW)
        # =====================================================
        current_tick = states[i]["tick"]
        audio_waveform = self._get_audio_window(
            sample["audio_path"],
            current_tick,
            sample["start_tick"]
        )

        # === 5. STATE ===
        st = states[i]

        state_vec = np.concatenate([
            np.array([
                st["hp"] / 100.0,
                st["armor"] / 100.0,
                float(st["helmet"]),
                self._safe_value(st["ammo"], 0.0) / 100.0,
                st["ct_alive"] / 5.0,
                st["t_alive"] / 5.0,
                self._safe_value(st["round_time_left"], 0.0) / 115.0,
                float(self._safe_value(st["bomb_planted"], 0.0)),
                float(self._safe_value(st["freeze_time"], 0.0)),
            ]),
            self._encode_side(st["side"]),
            self._encode_weapon(st["weapon"]),
            self._encode_weapon_list(st["weapon_list"]),
        ]).astype(np.float32)

        state_vec = torch.from_numpy(state_vec)

        # =====================================================
        # 6. HISTORICAL ACTIONS IN INTENT FORMAT
        # =====================================================
        intent_mouse_hist = []
        intent_keys_hist = []

        for k in range(self.actions_window):
            end = i - (k + 1) * T  # Сдвиг на 1 окно назад, чтобы не включать текущее окно (таргет)
            start = max(0, end - T + 1)
            if end < 0:
                break

            keys_window = set()
            mouse_start = None
            mouse_end = None

            for j in range(start, end + 1):
                st = states[j]

                if mouse_start is None:
                    mouse_start = np.array(st["mouse"], dtype=np.float32)
                mouse_end = np.array(st["mouse"], dtype=np.float32)

                keys_window.update(st["keys"])

            # Normalize mouse history same as target
            mouse_intent = (mouse_end - mouse_start) / self.MOUSE_SCALE

            window_intent = {}

            window_intent["fire"] = 1.0 if "MOUSE_LEFT" in keys_window else 0.0
            window_intent["second_fire"] = 1.0 if "MOUSE_RIGHT" in keys_window else 0.0

            window_intent["forward"] = 1.0 if "W" in keys_window else 0.0
            window_intent["back"] = 1.0 if "S" in keys_window else 0.0

            window_intent["left"] = 1.0 if ("A" in keys_window and "D" not in keys_window) else 0.0
            window_intent["right"] = 1.0 if ("D" in keys_window and "A" not in keys_window) else 0.0

            window_intent["jump"] = 1.0 if "SPACE" in keys_window else 0.0
            window_intent["crouch"] = 1.0 if "CTRL" in keys_window else 0.0
            window_intent["shift"] = 1.0 if "SHIFT" in keys_window else 0.0

            for key in ["WEAPON1", "WEAPON2", "WEAPON3", "C4", "R"]:
                window_intent[key.lower()] = 1.0 if key in keys_window else 0.0

            for key in ["HE", "MOLOTOV", "SMOKE", "FLASH", "DECOY"]:
                window_intent[key.lower()] = 1.0 if key in keys_window else 0.0

            window_intent["use"] = 1.0 if "E" in keys_window else 0.0

            intent_keys_hist.append(torch.tensor(list(window_intent.values()), dtype=torch.float32))
            intent_mouse_hist.append(torch.tensor(mouse_intent, dtype=torch.float32))

        # Паддинг нулями если истории не хватает (особенно в начале раунда)
        num_intent_keys = 20  # Количество клавиш в intent
        while len(intent_keys_hist) < self.actions_window:
            if len(intent_keys_hist) > 0:
                intent_keys_hist.append(torch.zeros_like(intent_keys_hist[0]))
            else:
                intent_keys_hist.append(torch.zeros(num_intent_keys, dtype=torch.float32))
        while len(intent_mouse_hist) < self.actions_window:
            intent_mouse_hist.append(torch.zeros(2, dtype=torch.float32))

        intent_keys_hist = torch.stack(intent_keys_hist[::-1])
        intent_mouse_hist = torch.stack(intent_mouse_hist[::-1])

        # =====================================================
        # 7. INTENT (aggregation over window T)
        # =====================================================
        target_keys_window = set()
        for j in range(t_start, i + 1):
            target_keys_window.update(states[j]["keys"])

        intent = {}

        intent["fire"] = 1.0 if "MOUSE_LEFT" in target_keys_window else 0.0
        intent["second_fire"] = 1.0 if "MOUSE_RIGHT" in target_keys_window else 0.0

        intent["forward"] = 1.0 if "W" in target_keys_window else 0.0
        intent["back"] = 1.0 if "S" in target_keys_window else 0.0

        intent["left"] = 1.0 if ("A" in target_keys_window and "D" not in target_keys_window) else 0.0
        intent["right"] = 1.0 if ("D" in target_keys_window and "A" not in target_keys_window) else 0.0

        intent["jump"] = 1.0 if "SPACE" in target_keys_window else 0.0
        intent["crouch"] = 1.0 if "CTRL" in target_keys_window else 0.0
        intent["shift"] = 1.0 if "SHIFT" in target_keys_window else 0.0

        for key in ["WEAPON1", "WEAPON2", "WEAPON3", "C4", "R"]:
            intent[key.lower()] = 1.0 if key in target_keys_window else 0.0

        for key in ["HE", "MOLOTOV", "SMOKE", "FLASH", "DECOY"]:
            intent[key.lower()] = 1.0 if key in target_keys_window else 0.0

        intent["use"] = 1.0 if "E" in target_keys_window else 0.0

        intent_vec = torch.tensor(list(intent.values()), dtype=torch.float32)

        # =====================================================
        # 8. TARGET: DELTA YAW / PITCH OVER WINDOW T (NORMALIZED)
        # =====================================================
        yaw_now = states[i]["mouse"][0]
        pitch_now = states[i]["mouse"][1]

        yaw_prev = states[t_start]["mouse"][0]
        pitch_prev = states[t_start]["mouse"][1]

        # Normalize by MOUSE_SCALE to bring values to ~[-1, 1] range
        # При инференсе: prediction * MOUSE_SCALE = actual delta
        target_mouse = torch.tensor(
            [(yaw_now - yaw_prev) / self.MOUSE_SCALE,
             (pitch_now - pitch_prev) / self.MOUSE_SCALE],
            dtype=torch.float32
        )

        # =====================================================
        return {
            "game_id": sample["game_id"],
            "round_id": sample["round_id"],
            "tick": states[i]["tick"],
            "scene_seq": scene_seq,
            "radar_seq": radar_seq,
            "audio_waveform": audio_waveform,  # NEW: (480000,)
            "actions_mouse": intent_mouse_hist,
            "actions_keys": intent_keys_hist,
            "state_vec": state_vec,
            "intent": intent_vec,
            "target_mouse": target_mouse,
            "T": i - t_start + 1
        }


class CSRoundDatasetNoAudio(CSRoundDataset):
    """Backward-compatible version without audio support."""

    def __init__(self, *args, **kwargs):
        kwargs['use_audio'] = False
        super().__init__(*args, **kwargs)
