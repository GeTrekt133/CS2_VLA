import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import random


class CSRoundDataset(Dataset):
    def __init__(
        self,
        dataset_json,
        T_min=3,
        T_max=12,
        radar_window=128,
        scene_window=16,
        actions_window=16,
        transform_scene=None,
        transform_radar=None,
        radar_crop_box=(10, 25, 140, 170),
        sampler=False
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

        # self.T_per_index = {}
        # for idx in range(len(self.samples)):
        #     self.T_per_index[idx] = random.choice([x for x in range(self.T_min, self.T_max + 1) if x % 4 == 0])
        
        self.allowed_T = [x for x in range(self.T_min, self.T_max + 1) if x % 4 == 0]


    # =====================================================
    def _load_metadata(self):
        with open(self.dataset_json, "r", encoding="utf-8") as f:
            full_meta = json.load(f)

        for game in full_meta["demos"]:
            demo_path = game["demo_path"]
            game_id = os.path.basename(demo_path)
            for rnd in game["rounds"]:
                self.metadata.append({
                    "game_id": game_id,
                    "round_id": rnd["round_id"],
                    "demo_path": demo_path,
                    "states": rnd["states"]
                })

    def _build_samples_index(self):
        for item in self.metadata:
            states = item["states"]
            for i in range(len(states)):
                if i % 4 == 0:  # якоримся как раньше
                    self.samples.append({
                        "game_id": item["game_id"],
                        "round_id": item["round_id"],
                        "demo_path": item["demo_path"],
                        "states": states,
                        "tick_idx": i
                    })

        print(f"[INFO] Prepared {len(self.samples)} intent samples")

    def __len__(self):
        # return 300
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _crop_radar(self, image):
        l, t, r, b = self.radar_crop_box
        img = image[t:b, l:r, :]
        img = cv2.resize(img, (224, 224))
        return img

    def _encode_keys(self, keys):
        vec = np.zeros(len(self.KEYS), dtype=np.float32)
        for k in keys:
            if k in self.KEYS:
                vec[self.KEYS.index(k)] = 1.0
        return vec


    def _encode_weapon(self, weapon):
        vec = np.zeros(len(self.WEAPONS), dtype=np.float32)
        w = weapon.lower()
        if w in self.WEAPONS:
            vec[self.WEAPONS.index(w)] = 1.0
        elif 'knife' in w or 'bayonet' in w or 'karambit' in w or 'daggers' in w:
            vec[self.WEAPONS.index("knife")] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def _encode_weapon_list(self, weapon_list):
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

    def _encode_side(self, side):
        vec = np.zeros(len(self.SIDES), dtype=np.float32)
        if side in self.SIDES:
            vec[self.SIDES.index(side)] = 1.0
        return vec
    
    def _safe_value(self, value, default=0.0):
        return float(value) if value is not None else default

    def _compute_T(self, idx):
        rng = random.Random(idx)
        return rng.choice(self.allowed_T)

    def __getitem__(self, idx):

        if self.sampling is True:
            i = self.samples[idx]["tick_idx"]
            T = self._compute_T(idx)

            scene_indices = list(range(max(0, i - (self.scene_window - 1) * T), i + 1, T))
            radar_indices = list(range(max(0, i - self.radar_window * 64 + 1), i + 1, 64))

            if radar_indices[-1] != i:
                radar_indices.append(i)
            ln_radar = len(radar_indices)
            ln_scene = len(scene_indices)
            # print("Lens: ", ln_scene, ln_radar)

            return ln_scene, ln_radar

        
        sample = self.samples[idx]
        states = sample["states"]
        demo_path = sample["demo_path"]
        i = sample["tick_idx"]

        # =====================================================
        # 1️⃣ INTENT WINDOW (рандомный T)
        # =====================================================
        T = self._compute_T(idx)
        t_start = max(0, i - T + 1)

        # =====================================================
        # 2️⃣ SCENE (шаг = T)
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
        # 3️⃣ RADAR (фиксированное окно, КАК БЫЛО)
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

        # === 4️⃣ СОСТОЯНИЕ ===
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
        # 4️⃣ HISTORICAL ACTIONS В ФОРМАТЕ INTENT
        # =====================================================
        intent_mouse_hist = []
        intent_keys_hist = []

        for k in range(self.actions_window):
            end = i - (k + 1) * T  # Сдвиг на 1 окно назад, чтобы не включать текущее окно (таргет)
            start = max(0, end - T + 1)
            if end < 0:
                break

            # объединяем нажатия мыши и клавиш в окне T
            keys_window = set()
            mouse_start = None
            mouse_end = None

            for j in range(start, end + 1):
                st = states[j]

                # сохраняем мышь для intent_mouse_hist (разница движения)
                if mouse_start is None:
                    mouse_start = np.array(st["mouse"], dtype=np.float32)
                mouse_end = np.array(st["mouse"], dtype=np.float32)

                # собираем все нажатия клавиш за окно
                keys_window.update(st["keys"])

            # mouse intent (разница)
            mouse_intent = mouse_end - mouse_start

            # keys intent в том же формате, что и таргет
            window_intent = {}

            # движение и основные действия
            window_intent["fire"] = 1.0 if "MOUSE_LEFT" in keys_window else 0.0
            window_intent["second_fire"] = 1.0 if "MOUSE_RIGHT" in keys_window else 0.0

            window_intent["forward"] = 1.0 if "W" in keys_window else 0.0
            window_intent["back"] = 1.0 if "S" in keys_window else 0.0

            # strafe → LEFT / RIGHT (бинарные)
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

            # конвертируем в тензор
            intent_keys_hist.append(torch.tensor(list(window_intent.values()), dtype=torch.float32))
            intent_mouse_hist.append(torch.tensor(mouse_intent, dtype=torch.float32))

        # дополняем нулями, если истории меньше actions_window (особенно в начале раунда)
        num_intent_keys = 20  # Количество клавиш в intent
        while len(intent_keys_hist) < self.actions_window:
            if len(intent_keys_hist) > 0:
                intent_keys_hist.append(torch.zeros_like(intent_keys_hist[0]))
            else:
                intent_keys_hist.append(torch.zeros(num_intent_keys, dtype=torch.float32))
        while len(intent_mouse_hist) < self.actions_window:
            intent_mouse_hist.append(torch.zeros(2, dtype=torch.float32))

        # формируем финальные тензоры
        intent_keys_hist = torch.stack(intent_keys_hist[::-1])   # (actions_window, intent_dim)
        intent_mouse_hist = torch.stack(intent_mouse_hist[::-1]) # (actions_window, 2)

        # =====================================================
        # 5️⃣ INTENT (агрегация по окну T)
        # =====================================================
        # Собираем все нажатия клавиш за окно [t_start, i]
        target_keys_window = set()
        for j in range(t_start, i + 1):
            target_keys_window.update(states[j]["keys"])

        intent = {}

        # движение и основные действия
        intent["fire"] = 1.0 if "MOUSE_LEFT" in target_keys_window else 0.0
        intent["second_fire"] = 1.0 if "MOUSE_RIGHT" in target_keys_window else 0.0

        intent["forward"] = 1.0 if "W" in target_keys_window else 0.0
        intent["back"] = 1.0 if "S" in target_keys_window else 0.0

        # strafe → LEFT / RIGHT (бинарные)
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

        # вектор таргета
        intent_vec = torch.tensor(list(intent.values()), dtype=torch.float32)

        # =====================================================
        # 6️⃣ TARGET: DELTA YAW / PITCH ПО ОКНУ T
        # =====================================================
        # print(states[i]["mouse"])
        yaw_now = states[i]["mouse"][0]
        pitch_now = states[i]["mouse"][1]

        yaw_prev = states[t_start]["mouse"][0]
        pitch_prev = states[t_start]["mouse"][1]

        target_mouse = torch.tensor(
            [yaw_now - yaw_prev, pitch_now - pitch_prev],
            dtype=torch.float32
        )
        # print(f"T - {T}, tick - {i}, T in sampler with same idx - {self.T_sampler[idx]}, idx - {idx}")
        # print(f"T - {T}, tick - {i}, idx - {idx}")
        # print(len(scene_indices))
        # print(scene_seq.shape, radar_seq.shape, intent_vec.shape, intent_mouse_hist.shape, intent_keys_hist.shape, target_mouse.shape)

        # =====================================================
        return {
            "game_id": sample["game_id"],
            "round_id": sample["round_id"],
            "tick": states[i]["tick"],
            "scene_seq": scene_seq,
            "radar_seq": radar_seq,
            "actions_mouse": intent_mouse_hist,
            "actions_keys": intent_keys_hist,
            "state_vec": state_vec,      
            "intent": intent_vec,        
            "target_mouse": target_mouse,
            "T": i - t_start + 1
        }
