"""
Alive digit inference module.

Crops two regions from a 640x640 frame and predicts CT/T alive counts (1-5).
Designed to run at 2 Hz (every 0.5s) alongside the main YOLO pipeline.

Usage:
    from alive_digit.inference import AliveDigitDetector

    detector = AliveDigitDetector('path/to/best.pt')
    ct_alive, t_alive = detector.predict(frame_640x640)
"""

import torch
import cv2
import numpy as np
from .model import AliveDigitCNN

# Crop coordinates in 640x640 frame
CT_CROP_CENTER = (288, 15)
T_CROP_CENTER = (350, 15)
CROP_SIZE = 25


class AliveDigitDetector:
    """Real-time alive count detector from screen capture."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = AliveDigitCNN()
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _crop_and_preprocess(self, frame: np.ndarray, center: tuple) -> torch.Tensor:
        """Crop digit region and convert to model input tensor."""
        cx, cy = center
        half = CROP_SIZE // 2
        h, w = frame.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + CROP_SIZE)
        y2 = min(h, y1 + CROP_SIZE)
        crop = frame[y1:y2, x1:x2]
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        tensor = torch.from_numpy(crop).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 25, 25)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> tuple:
        """
        Predict alive counts from a 640x640 frame.

        Args:
            frame: BGR or grayscale frame, 640x640
        Returns:
            (ct_alive, t_alive): tuple of ints (1-5)
        """
        ct_input = self._crop_and_preprocess(frame, CT_CROP_CENTER).to(self.device)
        t_input = self._crop_and_preprocess(frame, T_CROP_CENTER).to(self.device)
        # Batch both crops together
        batch = torch.cat([ct_input, t_input], dim=0)  # (2, 1, 25, 25)
        logits = self.model(batch)  # (2, 5)
        preds = logits.argmax(dim=1) + 1  # index 0-4 → label 1-5
        return int(preds[0]), int(preds[1])

    @torch.no_grad()
    def predict_with_confidence(self, frame: np.ndarray) -> tuple:
        """Same as predict but also returns confidence scores."""
        ct_input = self._crop_and_preprocess(frame, CT_CROP_CENTER).to(self.device)
        t_input = self._crop_and_preprocess(frame, T_CROP_CENTER).to(self.device)
        batch = torch.cat([ct_input, t_input], dim=0)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1) + 1
        confs = probs.max(dim=1).values
        return (int(preds[0]), float(confs[0])), (int(preds[1]), float(confs[1]))
