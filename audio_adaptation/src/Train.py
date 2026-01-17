"""
Training Script for CS2 AI Agent with Audio Support.

Updates:
- Added AudioEncoder for processing game audio
- Audio embeddings integrated into TemporalCrossTransformer
- Checkpoints include audio_encoder state
"""

import os
import cv2
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Any

from Dataloader import ExactMatchBucketSampler, debug_collate
from DatasetIntent import CSRoundDataset
from RadarEncoder import RadarEncoderEffB0
from Yolo import DetectionModel, Detect, load_pretrained_weights
from TemporalTransformer import TemporalCrossTransformer
from AudioEncoder import AudioEncoder
from sklearn.metrics import precision_score, recall_score


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_metrics(text: str, save_file: str):
    """Log metrics to file."""
    with open(save_file, "a") as f:
        f.write(text + "\n")


def detections(batch: Dict, model: nn.Module, max_det: int, batch_size: int, device: str, data_root: str):
    """Extract detections from YOLO model."""
    imgs = []
    for i in range(batch_size):
        tick = batch['tick'][i].item() - batch['tick'][i].item() % 4
        img_path = os.path.join(data_root, batch['game_id'][i], f"tick_{tick}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)

    imgs_batch = torch.from_numpy(np.stack(imgs)).to(device)
    with torch.no_grad():
        detects_raw = model(imgs_batch)[0][0]

    bboxes_batch = Detect.postprocess(detects_raw, 100, 80)

    detects_stack = []
    for i in range(batch_size):
        boxes_xyxy, conf, cls = bboxes_batch[i, :, :4], bboxes_batch[i, :, 4], bboxes_batch[i, :, 5]
        boxes_xyxy = ops.box_convert(boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy")
        keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.3)
        boxes_xyxy = boxes_xyxy[keep]
        conf = conf[keep]
        cls = cls[keep]

        stack = []
        for j, cl in enumerate(cls):
            if cl == 0:
                stack.append(torch.cat([boxes_xyxy[j] / 640, conf[j].unsqueeze(0)]))

        if len(stack) == 0:
            det_vec = torch.zeros(max_det * 5, device=device)
        else:
            det_vec = torch.cat(stack)
            n = det_vec.shape[0]
            if n > max_det * 5:
                det_vec = det_vec[:max_det * 5]
            else:
                det_vec = F.pad(det_vec, (0, max_det * 5 - n))

        detects_stack.append(det_vec.view(1, 1, max_det * 5))

    return torch.cat(detects_stack)


def evaluate(
    val_loader: DataLoader,
    radar_encoder: nn.Module,
    yolo: nn.Module,
    temporal_model: nn.Module,
    mse_loss: nn.Module,
    bce_loss: nn.Module,
    device: str,
    max_det: int,
    batch_size: int,
    w_keys: float,
    metrics_file: str,
    data_root: str,
    audio_encoder: Optional[nn.Module] = None,
    use_audio: bool = False
):
    """Evaluate model on validation set."""
    radar_encoder.eval()
    yolo.eval()
    temporal_model.eval()
    if audio_encoder is not None:
        audio_encoder.eval()

    total_loss = 0.0
    total_mse_mouse = 0.0

    all_preds_keys = []
    all_labels_keys = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if batch is None:
                print("val error")
                continue

            torch.cuda.empty_cache()

            scene_seq = batch["scene_seq"].to(device).permute(0, 1, 4, 2, 3)
            radar_seq = batch["radar_seq"].to(device).permute(0, 1, 4, 2, 3)
            actions_mouse = batch["actions_mouse"].to(device)
            actions_keys = batch["actions_keys"].to(device)
            state_vec = batch["state_vec"].to(device)
            gt_mouse = batch["target_mouse"].to(device)
            gt_keys = batch["intent"].to(device)

            detect_embeds = detections(batch, yolo, max_det, batch_size, device, data_root)

            # Radar encoding
            radar_embeds = torch.stack(
                [radar_encoder(radar_seq[:, t]) for t in range(radar_seq.shape[1])],
                dim=1
            )

            # Scene encoding
            scene_embeds = torch.stack(
                [yolo(scene_seq[:, t])[1] for t in range(scene_seq.shape[1])],
                dim=1
            )

            # Audio encoding (NEW)
            audio_embeds = None
            if use_audio and audio_encoder is not None and "audio_waveform" in batch:
                audio_waveform = batch["audio_waveform"].to(device)
                audio_embeds = audio_encoder(audio_waveform)  # (B, 60, 256)

            # Forward pass
            policy_mouse, policy_keys, _ = temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=torch.cat([actions_mouse, actions_keys], dim=-1),
                state_vec=state_vec,
                audio_seq=audio_embeds
            )

            # Loss
            loss_mouse = mse_loss(policy_mouse, gt_mouse)
            loss_keys = bce_loss(policy_keys, gt_keys)
            total_loss += (loss_mouse + w_keys * loss_keys).item()
            total_mse_mouse += loss_mouse.item()

            # Collect predictions
            preds_keys = torch.sigmoid(policy_keys).detach().cpu().numpy() > 0.5
            labels_keys = gt_keys.detach().cpu().numpy()

            all_preds_keys.append(preds_keys)
            all_labels_keys.append(labels_keys)

    # Aggregate metrics
    all_preds_keys = np.concatenate(all_preds_keys, axis=0)
    all_labels_keys = np.concatenate(all_labels_keys, axis=0)

    precision = precision_score(all_labels_keys.flatten(), all_preds_keys.flatten(), zero_division=0)
    recall = recall_score(all_labels_keys.flatten(), all_preds_keys.flatten(), zero_division=0)
    mse_mouse = total_mse_mouse / len(val_loader)
    avg_loss = total_loss / len(val_loader)

    log_text = f"Validation — Loss: {avg_loss:.4f} | MSE(mouse): {mse_mouse:.4f} | Precision(keys): {precision:.4f} | Recall(keys): {recall:.4f}"
    log_metrics(log_text, metrics_file)
    print(log_text)

    return total_loss / len(val_loader)


def train():
    """Main training function."""
    # ====== CONFIG ======
    device = "cuda"
    RUN_NAME = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    CKPT_ROOT = "./checkpoints2"
    DATA_ROOT = r"c:\Users\misas\CS2_NN\FramesDataset"  # Root directory for frame data
    TRAIN_DATASET_JSON = r"c:\Users\misas\CS2_NN\train_dataset.json"
    VAL_DATASET_JSON = r"c:\Users\misas\CS2_NN\val_dataset.json"
    AUDIO_DIR = r"D:\AudioData"  # Audio data directory (NEW)

    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    MAX_DET = 20
    LR = 1e-4
    W_KEYS = 100
    SEED = 42

    # Audio settings (NEW)
    USE_AUDIO = True
    AUDIO_EMBED_DIM = 512
    AUDIO_SEQ_LEN = 60

    # Paths
    CKPT_DIR = os.path.join(CKPT_ROOT, RUN_NAME, "checkpoints")
    LOG_DIR = os.path.join(CKPT_ROOT, RUN_NAME, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    METRICS_FILE = os.path.join(LOG_DIR, "metrics.txt")

    seed_everything(SEED)

    # ====== DATASETS ======
    train_dataset = CSRoundDataset(
        TRAIN_DATASET_JSON,
        use_audio=USE_AUDIO,
        audio_dir=AUDIO_DIR,
        audio_speedup_factor=4.0
    )
    val_dataset = CSRoundDataset(
        VAL_DATASET_JSON,
        use_audio=USE_AUDIO,
        audio_dir=AUDIO_DIR,
        audio_speedup_factor=4.0
    )

    train_sampler = ExactMatchBucketSampler(
        CSRoundDataset(TRAIN_DATASET_JSON, sampler=True, use_audio=False),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_sampler = ExactMatchBucketSampler(
        CSRoundDataset(VAL_DATASET_JSON, sampler=True, use_audio=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=debug_collate,
        batch_sampler=train_sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=debug_collate,
        batch_sampler=val_sampler,
        num_workers=0
    )

    # ====== MODELS ======
    # Load checkpoint if available
    checkpoint_path = None  # Set to path if resuming
    checkpoint = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Radar encoder
    radar_encoder = RadarEncoderEffB0(pretrained=True, in_ch=3, out_ch=64, embed_dim=512).to(device)
    if checkpoint and "radar_encoder" in checkpoint:
        radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=False)

    # YOLO
    yolo = DetectionModel(cfg="yolo11n.yaml").to(device)
    yolo = load_pretrained_weights(yolo, "yolo11n.pt").to(device).eval()
    if checkpoint and "yolo" in checkpoint:
        yolo.load_state_dict(checkpoint["yolo"], strict=False)

    # Audio encoder (NEW)
    audio_encoder = None
    if USE_AUDIO:
        audio_encoder = AudioEncoder(
            sample_rate=16000,
            embed_dim=AUDIO_EMBED_DIM,
            target_embeddings=AUDIO_SEQ_LEN,
            dropout=0.1
        ).to(device)
        if checkpoint and "audio_encoder" in checkpoint:
            audio_encoder.load_state_dict(checkpoint["audio_encoder"], strict=False)
        print(f"AudioEncoder initialized: {sum(p.numel() for p in audio_encoder.parameters()):,} params")

    # Temporal model (updated with audio support)
    temporal_model = TemporalCrossTransformer(
        use_audio=USE_AUDIO,
        audio_dim=AUDIO_EMBED_DIM,
        audio_seq=AUDIO_SEQ_LEN
    ).to(device)
    if checkpoint and "temporal_model" in checkpoint:
        temporal_model.load_state_dict(checkpoint["temporal_model"], strict=False)

    temporal_model = nn.DataParallel(temporal_model)

    # ====== LOSS & OPTIMIZER ======
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Collect parameters
    params = list(radar_encoder.parameters()) + list(yolo.parameters()) + list(temporal_model.parameters())
    if audio_encoder is not None:
        params += list(audio_encoder.parameters())

    optimizer = optim.AdamW(params, lr=LR)
    if checkpoint and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_val_loss = float("inf")
    start_epoch = checkpoint["epoch"] if checkpoint and "epoch" in checkpoint else 0

    # ====== TRAINING LOOP ======
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Set training mode
        radar_encoder.train()
        yolo.eval()
        temporal_model.train()
        if audio_encoder is not None:
            audio_encoder.train()

        # Freeze YOLO backbone, train only embeds
        for name, param in yolo.named_parameters():
            if "embeds" in name:
                param.requires_grad = True
            elif "model" in name:
                param.requires_grad = False

        total_loss = 0.0
        total_mse_mouse = 0.0
        all_preds_keys_train = []
        all_labels_keys_train = []

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            if batch is None:
                print("train error")
                continue

            torch.cuda.empty_cache()

            scene_seq = batch["scene_seq"].to(device).permute(0, 1, 4, 2, 3)
            radar_seq = batch["radar_seq"].to(device).permute(0, 1, 4, 2, 3)
            actions_mouse = batch["actions_mouse"].to(device)
            actions_keys = batch["actions_keys"].to(device)
            state_vec = batch["state_vec"].to(device)
            gt_mouse = batch["target_mouse"].to(device)
            gt_keys = batch["intent"].to(device)

            detect_embeds = detections(batch, yolo, MAX_DET, BATCH_SIZE, device, DATA_ROOT)

            # Radar encoding
            radar_embeds = torch.stack(
                [radar_encoder(radar_seq[:, t]) for t in range(radar_seq.shape[1])],
                dim=1
            )

            # Scene encoding
            scene_embeds = torch.stack(
                [yolo(scene_seq[:, t])[1] for t in range(scene_seq.shape[1])],
                dim=1
            )

            # Audio encoding (NEW)
            audio_embeds = None
            if USE_AUDIO and audio_encoder is not None and "audio_waveform" in batch:
                audio_waveform = batch["audio_waveform"].to(device)
                audio_embeds = audio_encoder(audio_waveform)  # (B, 60, 256)

            # Forward pass
            policy_mouse, policy_keys, _ = temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=torch.cat([actions_mouse, actions_keys], dim=-1),
                state_vec=state_vec,
                audio_seq=audio_embeds
            )

            # Loss
            loss_mouse = mse_loss(policy_mouse, gt_mouse)
            loss_keys = bce_loss(policy_keys, gt_keys)
            loss = loss_mouse + W_KEYS * loss_keys

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse_mouse += loss_mouse.item()

            # Collect predictions
            with torch.no_grad():
                preds_keys = torch.sigmoid(policy_keys).cpu().numpy() > 0.5
                labels_keys = gt_keys.cpu().numpy()
                all_preds_keys_train.append(preds_keys)
                all_labels_keys_train.append(labels_keys)

        # ====== TRAIN METRICS ======
        all_preds_keys_train = np.concatenate(all_preds_keys_train, axis=0)
        all_labels_keys_train = np.concatenate(all_labels_keys_train, axis=0)
        train_precision = precision_score(all_labels_keys_train.flatten(), all_preds_keys_train.flatten(), zero_division=0)
        train_recall = recall_score(all_labels_keys_train.flatten(), all_preds_keys_train.flatten(), zero_division=0)
        avg_train_mse_mouse = total_mse_mouse / len(train_loader)
        avg_train_loss = total_loss / len(train_loader)

        train_log = f"Train — Loss: {avg_train_loss:.4f} | MSE(mouse): {avg_train_mse_mouse:.4f} | Precision(keys): {train_precision:.4f} | Recall(keys): {train_recall:.4f}"
        log_metrics(train_log, METRICS_FILE)
        print(train_log)

        # ====== VALIDATION ======
        avg_val_loss = evaluate(
            val_loader, radar_encoder, yolo, temporal_model,
            mse_loss, bce_loss, device, MAX_DET, BATCH_SIZE, W_KEYS, METRICS_FILE, DATA_ROOT,
            audio_encoder=audio_encoder, use_audio=USE_AUDIO
        )

        log_text = f"Epoch [{epoch+1}/{NUM_EPOCHS}] — Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        log_metrics(log_text, METRICS_FILE)
        print(log_text)

        # ====== SAVE CHECKPOINT ======
        best_val_loss = avg_val_loss
        checkpoint_data = {
            "epoch": epoch + 1,
            "radar_encoder": radar_encoder.state_dict(),
            "yolo": yolo.state_dict(),
            "temporal_model": temporal_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        # Save audio encoder if used (NEW)
        if audio_encoder is not None:
            checkpoint_data["audio_encoder"] = audio_encoder.state_dict()

        torch.save(checkpoint_data, os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth"))

        print(f"Saved model with val_loss={avg_val_loss:.4f}")


if __name__ == "__main__":
    train()
