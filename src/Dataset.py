import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


class CSRoundDataset(Dataset):
    """
    Формирует обучающие сэмплы уровня 'тик' со скользящими окнами.
    Работает с единым dataset.json, содержащим все игры и раунды.
    """
    def __init__(self, dataset_json, sampler=False,
                 scene_window=16, radar_window=8192, action_window=192,
                 transform_scene=None, transform_radar=None,
                 radar_crop_box=(10,25,140,170)):
        self.dataset_json = dataset_json
        self.scene_window = scene_window
        self.radar_window = radar_window
        self.action_window = action_window
        self.transform_scene = transform_scene
        self.transform_radar = transform_radar
        self.radar_crop_box = radar_crop_box
        self.sampling = sampler

        self.metadata = []
        self.samples = []
        self.targets = []
        self._load_metadata()
        self._build_samples_index()

        self.KEYS = [
            "MOUSE_LEFT", "SPACE", "CTRL", "W", "S", "E", "A", "D", "MOUSE_RIGHT", "R", "MOLOTOV",
            "TAB", "F", "WEAPON1", "WEAPON2", "WEAPON3", "HE", "FLASH", "SMOKE", "DECOY", "C4", "SHIFT"
        ]

        self.WEAPONS = [
            'p2000', 'p250', 'five-seven', 'glock-18', 'tec-9', 'cz75-auto', 'dual berettas', 'desert eagle', 'm249',
            'r8 revolver', 'mp9', 'mac-10', 'pp-bizon', 'mp7', 'ump-45', 'p90', 'mp5-sd', 'famas', 'galil ar', 'sawed-off',
            'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553', 'ssg 08', 'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
            'negev', 'knife', 'high explosive grenade', 'flashbang', 'smoke grenade', 'decoy grenade', 'molotov', 
            'incendiary grenade', 'c4 explosive', 'None'
        ]

        self.SIDES = ["CT", "T"]

    # -----------------------------------
    def _load_metadata(self):
        """Читаем единый dataset.json"""
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
                    "data": rnd
                })
        self.metadata.sort(key=lambda x: (x["game_id"], x["round_id"]))

    # -----------------------------------
    def _build_samples_index(self):
        """Создаём список всех (игра, раунд, индекс тика)"""
        for item in self.metadata:
            states = item["data"]["states"]
            for i in range(len(states)):
                res_dict = {
                    "game_id": item["game_id"],
                    "start_tick": item["data"]["start_tick"],
                    "end_tick": item["data"]["end_tick"],
                    "round_id": item["round_id"],
                    "demo_path": item["demo_path"],
                    "states": states,
                    "tick_idx": i
                }
                self.targets.append(res_dict)
                if i % 4 == 0:
                    self.samples.append(res_dict)
        print(f"[INFO] Prepared {len(self.samples)} tick-level samples")

    # -----------------------------------
    def __len__(self):
        # return 500
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _crop_radar(self, image: np.ndarray):
        left, top, right, bottom = self.radar_crop_box
        img = image[top:bottom, left:right, :]
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
            vec[self.WEAPONS.index('knife')] = 1.0
        else:
            vec[-1] = 1.0  # "None"
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

    # -----------------------------------
    def __getitem__(self, idx):
        if self.sampling == True:
            i = self.samples[idx]["tick_idx"]
            start = max(0, i - (self.scene_window - 1) * 4)
            ln_scene = len(list(range(start, i + 1, 4)))
            radar_indices = list(range(max(0, i - self.radar_window + 1), i + 1, 64))
            ln_radar = len(radar_indices)
            if radar_indices[-1] != i:
                ln_radar += 1
            return ln_scene, ln_radar
        sample = self.samples[idx]
        states = sample["states"]
        demo_path = sample["demo_path"]
        i = sample["tick_idx"]

        # === 1️⃣ СЦЕНЫ (последние 16 тиков, шаг 4) ===
        start = max(0, i - (self.scene_window - 1) * 4)
        scene_indices = list(range(start, i + 1, 4))
        scene_frames = []
        for j in scene_indices:
            tick = sample["start_tick"] + j
            frame_path = os.path.join(demo_path, f"tick_{tick}.jpg")
            if os.path.exists(frame_path):
                img = self._load_image(frame_path)
                if self.transform_scene:
                    img = self.transform_scene(img)
                scene_frames.append(img)
        if len(scene_frames) == 0:
            scene_frames.append(np.zeros((640, 640, 3), dtype=np.float32))
        scene_seq = torch.tensor(np.stack(scene_frames), dtype=torch.float32)

        # radar
        radar_indices = list(range(max(0, i - self.radar_window + 1), i + 1, 64))
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

        # === 3️⃣ ДЕЙСТВИЯ (192 последних из self.targets, не включая текущий) ===
        actions_mouse = []
        actions_keys = []
        for j in range(max(0, i - self.action_window), i):
            st = self.targets[j]["states"][self.targets[j]["tick_idx"]]
            mouse = np.array(st["mouse"], dtype=np.float32)
            keys_vec = self._encode_keys(st["keys"])
            actions_mouse.append(mouse)
            actions_keys.append(keys_vec)
        if len(actions_mouse) < self.action_window:
            pad_len = self.action_window - len(actions_mouse)
            actions_mouse = [np.zeros(2, dtype=np.float32)] * pad_len + actions_mouse
            actions_keys = [np.zeros(22, dtype=np.float32)] * pad_len + actions_keys
        actions_mouse = torch.tensor(np.stack(actions_mouse), dtype=torch.float32)
        actions_keys = torch.tensor(np.stack(actions_keys), dtype=torch.float32)

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
                float(self._safe_value(st["freeze_time"], 0.0))
            ]),
            np.array(self._encode_side(st["side"])),
            np.array(self._encode_weapon(st["weapon"])),
            np.array(self._encode_weapon_list(st["weapon_list"]))
        ])
        state_vec = torch.tensor(state_vec, dtype=torch.float32)

        # === 5️⃣ ЛЕЙБЛЫ (текущий тик + 3 следующих из self.targets) ===
        label_mouse = []
        label_keys = []
        for j in range(i, i + 4):
            if j < len(self.targets):
                st = self.targets[j]["states"][self.targets[j]["tick_idx"]]
                mouse = np.array(st["mouse"], dtype=np.float32)
                keys_vec = self._encode_keys(st["keys"])
                label_mouse.append(mouse)
                label_keys.append(keys_vec)
            else:
                label_mouse.append(np.zeros(2, dtype=np.float32))
                label_keys.append(np.zeros(22, dtype=np.float32))
        label_mouse = torch.tensor(np.stack(label_mouse), dtype=torch.float32)
        label_keys = torch.tensor(np.stack(label_keys), dtype=torch.float32)
        # print(label_keys.shape)

        return {
            "game_id": sample["game_id"],
            "round_id": sample["round_id"],
            "tick": st["tick"],
            "scene_seq": scene_seq,       # (T, C, H, W)
            "radar_seq": radar_seq,       # (T, C, H_r, W_r)
            "actions_mouse": actions_mouse,   # "mouse": (192, 2)
            "actions_keys": actions_keys,   # "keys": (192, 22)
            "state_vec": state_vec,       # (9,)
            "labels_mouse": label_mouse,     # (4, 2)
            "labels_keys": label_keys        # (4, 10)
        }
        
