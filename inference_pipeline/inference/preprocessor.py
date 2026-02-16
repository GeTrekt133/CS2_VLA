"""
Preprocessor for preparing model inputs from raw capture data.
Handles normalization, tensor conversion, and sequence preparation.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch


class Preprocessor:
    """
    Prepares raw capture data for model inference.

    Converts numpy arrays to properly shaped/normalized tensors:
    - Scene frames: (B, T, 3, H, W) normalized [0, 1]
    - Radar frames: (B, T, 3, 224, 224) normalized [0, 1]
    - Audio: (B, 480000) -> AudioEncoder -> (B, 60, 512)
    - State: (B, 95)
    - Actions: (B, 16, 22)
    """

    def __init__(
        self,
        device: str = "cuda",
        radar_seq_len: int = 32,  # Reduced from 129 for better perf
        scene_seq_len: int = 16,
        action_seq_len: int = 16,
        state_dim: int = 95,
    ):
        """
        Initialize preprocessor.

        Args:
            device: Target device for tensors
            radar_seq_len: Number of radar frames (129)
            scene_seq_len: Number of scene frames (16)
            action_seq_len: Number of action history entries (16)
            state_dim: State vector dimension (95)
        """
        self.device = device
        self.radar_seq_len = radar_seq_len
        self.scene_seq_len = scene_seq_len
        self.action_seq_len = action_seq_len
        self.state_dim = state_dim

    def prepare_scene_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]],
        target_size: Tuple[int, int] = (640, 480),
    ) -> torch.Tensor:
        """
        Prepare scene frames for model input.

        Args:
            frames: List of (timestamp, frame) where frame is (H, W, 3) uint8 or float
            target_size: (width, height) for resizing if needed

        Returns:
            Tensor (1, scene_seq_len, 3, H, W) normalized [0, 1]
        """
        if len(frames) == 0:
            # Return zeros if no frames
            return torch.zeros(
                1, self.scene_seq_len, 3, target_size[1], target_size[0],
                dtype=torch.float32, device=self.device
            )

        # Extract frames
        frame_list = []
        for ts, frame in frames[-self.scene_seq_len:]:
            # Normalize to [0, 1] if needed
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            elif frame.max() > 1.0:
                frame = frame / 255.0

            # Convert (H, W, C) -> (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            frame_list.append(frame)

        # Pad if not enough frames
        while len(frame_list) < self.scene_seq_len:
            frame_list.insert(0, frame_list[0] if frame_list else np.zeros((3, target_size[1], target_size[0])))

        # Stack to (T, C, H, W)
        sequence = np.stack(frame_list, axis=0)

        # Convert to tensor (1, T, C, H, W)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        return tensor

    def prepare_radar_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]],
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Prepare radar frames for model input.

        Args:
            frames: List of (timestamp, frame) where frame is (H, W, 3)
            target_size: Target size (should already be 224x224)

        Returns:
            Tensor (1, radar_seq_len, 3, 224, 224) normalized [0, 1]
        """
        if len(frames) == 0:
            return torch.zeros(
                1, self.radar_seq_len, 3, target_size[1], target_size[0],
                dtype=torch.float32, device=self.device
            )

        frame_list = []
        for ts, frame in frames[-self.radar_seq_len:]:
            # Normalize
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            elif frame.max() > 1.0:
                frame = frame / 255.0

            # (H, W, C) -> (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            frame_list.append(frame)

        # Pad if needed
        while len(frame_list) < self.radar_seq_len:
            frame_list.insert(0, frame_list[0] if frame_list else np.zeros((3, target_size[1], target_size[0])))

        sequence = np.stack(frame_list, axis=0)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        return tensor

    def prepare_audio(
        self,
        audio_buffer: np.ndarray,
        expected_samples: int = 480000,
    ) -> torch.Tensor:
        """
        Prepare audio buffer for AudioEncoder.

        Args:
            audio_buffer: Raw audio (N,) float32
            expected_samples: Expected number of samples (30 sec @ 16kHz)

        Returns:
            Tensor (1, expected_samples) for AudioEncoder
        """
        if len(audio_buffer) < expected_samples:
            # Pad with zeros at the beginning
            padding = np.zeros(expected_samples - len(audio_buffer), dtype=np.float32)
            audio_buffer = np.concatenate([padding, audio_buffer])
        elif len(audio_buffer) > expected_samples:
            # Take the most recent samples
            audio_buffer = audio_buffer[-expected_samples:]

        # Normalize
        max_val = np.abs(audio_buffer).max()
        if max_val > 0:
            audio_buffer = audio_buffer / max_val * 0.95

        tensor = torch.from_numpy(audio_buffer).unsqueeze(0).to(self.device)
        return tensor

    def prepare_state_vector(
        self,
        state_vec: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Prepare state vector from GSI data.

        Args:
            state_vec: State vector (95,) or None

        Returns:
            Tensor (1, 95)
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
            action_history: (action_seq_len, 22) combined mouse + keys

        Returns:
            Tensor (1, action_seq_len, 22)
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
        detections: Optional[np.ndarray],
        max_det: int = 20,
        features_per_det: int = 5,
    ) -> torch.Tensor:
        """
        Prepare detection tensor.

        Args:
            detections: (N, 5) detections [x, y, w, h, conf] or None
            max_det: Maximum detections (20)
            features_per_det: Features per detection (5)

        Returns:
            Tensor (1, 1, 100) flattened detections
        """
        total_dim = max_det * features_per_det  # 100

        if detections is None or len(detections) == 0:
            return torch.zeros(1, 1, total_dim, dtype=torch.float32, device=self.device)

        # Pad or truncate to max_det
        if len(detections) < max_det:
            padding = np.zeros((max_det - len(detections), features_per_det), dtype=np.float32)
            detections = np.vstack([detections, padding])
        else:
            detections = detections[:max_det]

        # Flatten to (100,)
        flat = detections.flatten().astype(np.float32)

        tensor = torch.from_numpy(flat).view(1, 1, total_dim).to(self.device)
        return tensor

    def encode_radar_sequence(
        self,
        radar_encoder: torch.nn.Module,
        radar_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode radar frames through RadarEncoder.

        Args:
            radar_encoder: RadarEncoderEffB0 model
            radar_frames: (1, T, 3, 224, 224)

        Returns:
            (1, T, 512) radar embeddings
        """
        B, T = radar_frames.shape[:2]

        # Flatten batch and time dimensions
        flat = radar_frames.view(B * T, 3, 224, 224)

        with torch.no_grad():
            embeds_flat = radar_encoder(flat)  # (B*T, 512)

        # Reshape back
        embeds = embeds_flat.view(B, T, -1)

        return embeds

    def encode_scene_sequence(
        self,
        yolo_model: torch.nn.Module,
        scene_frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode scene frames through YOLO and get both embeddings and detections.

        Args:
            yolo_model: DetectionModel
            scene_frames: (1, T, 3, H, W)

        Returns:
            (scene_embeds, detections):
            - scene_embeds: (1, T, 2048)
            - detections: (1, 1, 100) from latest frame only
        """
        B, T = scene_frames.shape[:2]
        H, W = scene_frames.shape[3:]

        # Flatten
        flat = scene_frames.view(B * T, 3, H, W)

        with torch.no_grad():
            # YOLO returns (detections, scene_embed)
            det_output, scene_embeds_flat = yolo_model(flat)

        # Reshape scene embeddings
        scene_embeds = scene_embeds_flat.view(B, T, -1)

        # For detections, we only use the latest frame
        # TODO: Add proper detection post-processing if needed
        # For now, return zeros as placeholder
        detections = torch.zeros(B, 1, 100, device=self.device)

        return scene_embeds, detections

    def encode_audio(
        self,
        audio_encoder: torch.nn.Module,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode audio through AudioEncoder.

        Args:
            audio_encoder: AudioEncoder model
            audio: (1, 480000) waveform

        Returns:
            (1, 60, 512) audio embeddings
        """
        with torch.no_grad():
            embeds = audio_encoder(audio)

        return embeds


def test_preprocessor():
    """Test preprocessor functionality."""
    print("Testing Preprocessor...")

    prep = Preprocessor(device="cpu")

    # Test scene preparation
    frames = [(0.0, np.random.rand(480, 640, 3).astype(np.float32)) for _ in range(10)]
    scene = prep.prepare_scene_sequence(frames)
    print(f"Scene shape: {scene.shape}")  # (1, 16, 3, 480, 640)

    # Test radar preparation
    radar_frames = [(0.0, np.random.rand(224, 224, 3).astype(np.float32)) for _ in range(50)]
    radar = prep.prepare_radar_sequence(radar_frames)
    print(f"Radar shape: {radar.shape}")  # (1, 129, 3, 224, 224)

    # Test audio preparation
    audio = np.random.rand(300000).astype(np.float32)
    audio_tensor = prep.prepare_audio(audio)
    print(f"Audio shape: {audio_tensor.shape}")  # (1, 480000)

    # Test state preparation
    state = np.random.rand(95).astype(np.float32)
    state_tensor = prep.prepare_state_vector(state)
    print(f"State shape: {state_tensor.shape}")  # (1, 95)

    # Test action history
    actions = np.random.rand(16, 22).astype(np.float32)
    action_tensor = prep.prepare_action_history(actions)
    print(f"Actions shape: {action_tensor.shape}")  # (1, 16, 22)

    # Test detections
    dets = np.random.rand(10, 5).astype(np.float32)
    det_tensor = prep.prepare_detections(dets)
    print(f"Detections shape: {det_tensor.shape}")  # (1, 1, 100)

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_preprocessor()
