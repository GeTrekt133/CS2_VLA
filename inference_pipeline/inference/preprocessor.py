"""
Preprocessor for preparing model inputs from raw capture data.
Handles normalization, tensor conversion, and sequence preparation.

Aligned with final_model/ architecture:
  - Scene: (1, 64, 3, 640, 640) → YOLO embed → (1, 64, 512) for ModalityCompressor
  - Radar: (1, 16, 3, 224, 224) → RadarEncoder → (1, 16, 512)
  - Audio: (1, 2, 256000) stereo → StereoAudioEncoder → (1, 32, 512) → linspace → (1, 16, 512)
  - State: (1, 100)
  - Actions: (1, 64, 22)
  - Detection: (1, 16, 100)
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


class Preprocessor:
    """
    Prepares raw capture data for model inference.

    Converts numpy arrays to properly shaped/normalized tensors matching
    final_model/ TemporalCrossTransformer input format.
    """

    def __init__(
        self,
        device: str = "cuda",
        radar_seq_len: int = 16,     # model expects 16 radar tokens
        scene_seq_len: int = 64,     # model expects 64 scene tokens (compressed to 16)
        action_seq_len: int = 64,    # model expects 64 action tokens (compressed to 16)
        state_dim: int = 100,        # 12 scalars + 2 side + 43 weapon + 43 weapon_list
    ):
        self.device = device
        self.radar_seq_len = radar_seq_len
        self.scene_seq_len = scene_seq_len
        self.action_seq_len = action_seq_len
        self.state_dim = state_dim

    def prepare_scene_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]],
        target_size: Tuple[int, int] = (640, 640),
    ) -> torch.Tensor:
        """
        Prepare scene frames for YOLO input.

        Args:
            frames: List of (timestamp, frame) where frame is (H, W, 3) uint8 or float
            target_size: (width, height) — must be 640×640 for YOLO

        Returns:
            Tensor (1, scene_seq_len, 3, H, W) normalized [0, 1]
        """
        if len(frames) == 0:
            return torch.zeros(
                1, self.scene_seq_len, 3, target_size[1], target_size[0],
                dtype=torch.float32, device=self.device
            )

        frame_list = []
        for ts, frame in frames[-self.scene_seq_len:]:
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            elif frame.max() > 1.0:
                frame = frame / 255.0

            # Resize to 640×640 if needed
            if frame.shape[0] != target_size[1] or frame.shape[1] != target_size[0]:
                import cv2
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

            # (H, W, C) -> (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            frame_list.append(frame)

        # Pad if not enough frames (repeat first frame)
        while len(frame_list) < self.scene_seq_len:
            frame_list.insert(0, frame_list[0] if frame_list else np.zeros((3, target_size[1], target_size[0])))

        sequence = np.stack(frame_list, axis=0)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        return tensor

    def prepare_radar_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]],
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Prepare radar frames for RadarEncoder input.

        If more frames than radar_seq_len, linspace-selects to get exactly radar_seq_len.

        Args:
            frames: List of (timestamp, frame) where frame is (H, W, 3)

        Returns:
            Tensor (1, radar_seq_len, 3, 224, 224) normalized [0, 1]
        """
        if len(frames) == 0:
            return torch.zeros(
                1, self.radar_seq_len, 3, target_size[1], target_size[0],
                dtype=torch.float32, device=self.device
            )

        # Linspace select if we have more frames than needed
        if len(frames) > self.radar_seq_len:
            indices = np.linspace(0, len(frames) - 1, self.radar_seq_len).astype(int)
            frames = [frames[i] for i in indices]

        frame_list = []
        for ts, frame in frames[-self.radar_seq_len:]:
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            elif frame.max() > 1.0:
                frame = frame / 255.0

            frame = np.transpose(frame, (2, 0, 1))
            frame_list.append(frame)

        while len(frame_list) < self.radar_seq_len:
            frame_list.insert(0, frame_list[0] if frame_list else np.zeros((3, target_size[1], target_size[0])))

        sequence = np.stack(frame_list, axis=0)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        return tensor

    def prepare_audio(
        self,
        audio_buffer: np.ndarray,
        expected_samples: int = 256000,
    ) -> torch.Tensor:
        """
        Prepare stereo audio buffer for StereoAudioEncoder.

        Args:
            audio_buffer: (2, N) stereo float32 or (N,) mono
            expected_samples: Expected samples per channel (16 sec @ 16kHz = 256000)

        Returns:
            Tensor (1, 2, expected_samples) for StereoAudioEncoder
        """
        # Normalize to (2, N)
        if audio_buffer.ndim == 1:
            # Mono fallback: duplicate to stereo
            audio_buffer = np.stack([audio_buffer, audio_buffer], axis=0)
        elif audio_buffer.ndim == 2 and audio_buffer.shape[0] != 2:
            if audio_buffer.shape[1] == 2:
                audio_buffer = audio_buffer.T
            else:
                audio_buffer = np.stack([audio_buffer[0], audio_buffer[0]], axis=0)

        # Pad or truncate to expected_samples
        n_samples = audio_buffer.shape[1]
        if n_samples < expected_samples:
            padding = np.zeros((2, expected_samples - n_samples), dtype=np.float32)
            audio_buffer = np.concatenate([padding, audio_buffer], axis=1)
        elif n_samples > expected_samples:
            audio_buffer = audio_buffer[:, -expected_samples:]

        # Normalize per-channel
        max_val = np.abs(audio_buffer).max()
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.95

        tensor = torch.from_numpy(audio_buffer.astype(np.float32)).unsqueeze(0).to(self.device)
        return tensor  # (1, 2, 256000)

    def prepare_state_vector(
        self,
        state_vec: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Prepare state vector from GSI data.

        Args:
            state_vec: (100,) or None

        Returns:
            Tensor (1, 100)
        """
        if state_vec is None:
            return torch.zeros(1, self.state_dim, dtype=torch.float32, device=self.device)

        tensor = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(self.device)
        return tensor

    def prepare_action_history(
        self,
        action_history: np.ndarray,
    ) -> torch.Tensor:
        """
        Prepare action history for model input.

        Args:
            action_history: (64, 22) combined mouse + keys

        Returns:
            Tensor (1, 64, 22)
        """
        if action_history is None:
            return torch.zeros(
                1, self.action_seq_len, 22,
                dtype=torch.float32, device=self.device
            )

        tensor = torch.from_numpy(action_history.astype(np.float32)).unsqueeze(0).to(self.device)
        return tensor

    def prepare_detections(
        self,
        detection_sequence: np.ndarray,
    ) -> torch.Tensor:
        """
        Prepare detection sequence tensor.

        Args:
            detection_sequence: (16, 100) detection vectors from DetectionBuffer

        Returns:
            Tensor (1, 16, 100)
        """
        if detection_sequence is None:
            return torch.zeros(1, 16, 100, dtype=torch.float32, device=self.device)

        tensor = torch.from_numpy(detection_sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        return tensor

    def encode_radar_sequence(
        self,
        radar_encoder: torch.nn.Module,
        radar_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode radar frames through RadarEncoder.

        Args:
            radar_encoder: RadarEncoderEffB0
            radar_frames: (1, T, 3, 224, 224)

        Returns:
            (1, T, 512) radar embeddings
        """
        B, T = radar_frames.shape[:2]

        flat = radar_frames.view(B * T, 3, 224, 224)

        with torch.no_grad():
            embeds_flat = radar_encoder(flat)  # (B*T, 512)

        embeds = embeds_flat.view(B, T, -1)
        return embeds

    def encode_scene_frame(
        self,
        yolo_model: torch.nn.Module,
        scene_frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single scene frame through YOLO.
        Returns both scene embedding and detection vector.

        Args:
            yolo_model: DetectionModel (final_model/Yolo.py)
            scene_frame: (1, 3, 640, 640)

        Returns:
            (scene_embed, det_vec):
            - scene_embed: (1, 512)
            - det_vec: (100,) flattened detections
        """
        with torch.no_grad():
            # extract_features: stops at layer 16, returns (p3, p5)
            p3, p5 = yolo_model.extract_features(scene_frame)

        # embed_from_features: at inference, no_grad is fine
        with torch.no_grad():
            scene_embed = yolo_model.embed_from_features(p3, p5)  # (1, 512)

        # For detections, run full forward
        with torch.no_grad():
            det_raw = yolo_model.forward_detections_only(scene_frame)

        # Postprocess detections
        det_vec = self._postprocess_detections(det_raw, yolo_model, max_det=20)

        return scene_embed, det_vec

    def _postprocess_detections(
        self,
        det_raw: torch.Tensor,
        yolo_model: torch.nn.Module,
        max_det: int = 20,
    ) -> np.ndarray:
        """
        Postprocess YOLO raw output to detection vector (100,).

        Args:
            det_raw: Raw YOLO output (decoded preds or tuple)
            yolo_model: DetectionModel for Detect.postprocess
            max_det: Max detections (20)

        Returns:
            (100,) detection vector [20 × (cx, cy, w, h, conf)]
        """
        try:
            # det_raw from forward_detections_only in eval mode: (decoded, raw_feats) tuple
            if isinstance(det_raw, tuple):
                decoded = det_raw[0]  # (B, 4+nc, N_anchors)
            else:
                decoded = det_raw

            # Use Detect.postprocess to get (B, max_det, 6)
            from Yolo import Detect
            post = Detect.postprocess(decoded, max_det=max_det, nc=2)
            # post: (B, max_det, 6) [cx, cy, w, h, conf, cls]

            # Take first batch, first 5 features (cx, cy, w, h, conf)
            dets = post[0, :, :5].cpu().numpy()  # (max_det, 5)

            # Normalize coordinates to [0, 1] (assuming 640×640 input)
            dets[:, :4] /= 640.0

            return dets.flatten().astype(np.float32)  # (100,)

        except Exception:
            return np.zeros(max_det * 5, dtype=np.float32)

    def encode_audio(
        self,
        audio_encoder: torch.nn.Module,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode stereo audio through StereoAudioEncoder + linspace to 16.

        Args:
            audio_encoder: StereoAudioEncoder
            audio: (1, 2, 256000) stereo waveform

        Returns:
            (1, 16, 512) audio embeddings (linspaced from 32)
        """
        with torch.no_grad():
            embeds = audio_encoder(audio)  # (1, 32, 512)

        # Linspace 32 → 16 to match unified SEQ_LEN
        if embeds.shape[1] > 16:
            # Interpolate: (1, 32, 512) → (1, 16, 512)
            embeds = embeds.permute(0, 2, 1)  # (1, 512, 32)
            embeds = F.interpolate(embeds, size=16, mode='linear', align_corners=True)
            embeds = embeds.permute(0, 2, 1)  # (1, 16, 512)

        return embeds
