"""
Training Script — Final 50M Architecture CS2 AI Agent.

Architecture:
  RadarEncoder (EffB0 blocks 1-5): ~1.0M
  YOLO (YOLOv11l backbone):       ~25.4M  (frozen, only embeds trainable)
  YOLO embeds head:                ~1.6M  (trainable)
  AudioEncoder (4-block QuartzNet):~3.5M
  TemporalTransformer (d=384,L=4): ~19M
  FlowActionHead:                  ~0.3M
  ─────────────────────────────────────
  Total: ~50.8M  |  Perception/Control: 1.66x ✅

Sequence inputs (all unified to SEQ_LEN=16 tokens):
  radar_seq:     32 raw @1Hz    → linspace → (B, 16, 512)
  scene_seq:     64 raw @~16Hz  → YOLO embed → (B, 16, 512)
  audio_seq:     32 raw @0.5s   → AudioEncoder(32) → linspace → (B, 16, 512)
  detection_seq: 64 raw @~16Hz  → YOLO detect → linspace → (B, 16, 100)
  action_seq:    dataset gives (16, 22) directly (linspace done in DatasetIntent)
  state_vec:     (B, 95) latest observation

Backbone caching:
  P3 features (layer 16 of YOLO) are cached per frame path (float16, CPU).
  Detection vectors are cached per frame path (float32, CPU).
  embeds head always runs with gradient.

Loss:
  flow_loss: FlowMatching for mouse (yaw, pitch)
  bce_loss:  BCEWithLogitsLoss for 20 key actions  (weight W_KEYS)
"""

import os
import cv2
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from collections import OrderedDict

from DatasetIntent import CSRoundDataset, SEQ_LEN, SCENE_SEQ_LEN, ACTION_SEQ_LEN
from RadarEncoder import RadarEncoderEffB0
from Yolo import DetectionModel, Detect, load_pretrained_weights
from TemporalTransformer import TemporalCrossTransformer, FlowActionHead
from AudioEncoder import AudioEncoder

# sklearn is optional for metrics
try:
    from sklearn.metrics import precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# Config
# ============================================================================

KEY_NAMES = [
    'fire', 'second_fire', 'forward', 'back', 'left', 'right',
    'jump', 'crouch', 'shift', 'weapon1', 'weapon2', 'weapon3',
    'c4', 'reload', 'he', 'molotov', 'smoke', 'flash', 'decoy', 'use',
]

# ============================================================================
# Helpers
# ============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_metrics(text: str, save_file: str):
    with open(save_file, 'a') as f:
        f.write(text + '\n')


def compute_per_class_metrics(all_labels: np.ndarray, all_preds: np.ndarray) -> Dict:
    metrics = {}
    for i in range(all_labels.shape[1]):
        lbl = all_labels[:, i]
        prd = all_preds[:, i]
        tp = np.sum((lbl == 1) & (prd == 1))
        fp = np.sum((lbl == 0) & (prd == 1))
        fn = np.sum((lbl == 1) & (prd == 0))
        support = int(tp + fn)
        pred_pos = int(tp + fp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        key = KEY_NAMES[i] if i < len(KEY_NAMES) else f'key_{i}'
        metrics[key] = {
            'precision': float(prec), 'recall': float(rec),
            'support': support, 'predicted_positives': pred_pos,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        }
    return metrics


# ============================================================================
# Backbone utilities
# ============================================================================

def load_image_tensor(path: str, img_size: int = 640) -> torch.Tensor:
    """Load BGR image from path → (3, img_size, img_size) float tensor [0,1]."""
    img = cv2.imread(path)
    if img is None:
        return torch.zeros(3, img_size, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def get_backbone_features_batch(
    frame_paths: List[str],
    yolo: DetectionModel,
    device: str,
    img_size: int = 640,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get P3 (80×80) and backbone P5 (20×20) features for a list of frame paths.

    Returns:
        p3_batch: (N, p3_ch, 80, 80) float32 on device
        p5_batch: (N, p5_ch, 20, 20) float32 on device
    """
    imgs = torch.stack([load_image_tensor(p, img_size) for p in frame_paths]).to(device)
    p3_batch, p5_batch = yolo.extract_features(imgs)
    return p3_batch.float(), p5_batch.float()


@torch.no_grad()
def get_det_vector(frame_path: str, yolo: DetectionModel,
                   device: str,
                   max_det: int = 20, img_size: int = 640,
                   nc: int = 2) -> torch.Tensor:
    """
    Get detection vector (max_det * 5,) for a single frame.
    Detection = [x1,y1,x2,y2,conf] per body bbox.
    """
    img = load_image_tensor(frame_path, img_size).unsqueeze(0).to(device)
    raw = yolo.forward_detections_only(img)   # full YOLO forward, no embed

    # Decode
    if isinstance(raw, tuple):
        decoded = raw[0]  # eval mode: (decoded, raw_feats)
    else:
        decoded = raw

    bboxes_batch = Detect.postprocess(decoded, max_det, nc)  # (1, max_det, 6)
    bboxes = bboxes_batch[0]   # (max_det, 6) [cx,cy,w,h,conf,cls]

    boxes_xyxy = ops.box_convert(bboxes[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
    conf = bboxes[:, 4]
    cls = bboxes[:, 5]
    keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.3)

    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]

    stack = []
    for j, cl in enumerate(cls):
        if int(cl.item()) == 0:  # body only
            stack.append(torch.cat([boxes_xyxy[j] / img_size, conf[j].unsqueeze(0)]))

    if len(stack) == 0:
        det_vec = torch.zeros(max_det * 5, device=device)
    else:
        det_vec = torch.cat(stack)
        n = det_vec.shape[0]
        if n > max_det * 5:
            det_vec = det_vec[:max_det * 5]
        else:
            det_vec = F.pad(det_vec, (0, max_det * 5 - n))

    return det_vec.float()


def build_detection_sequence(
    batch: Dict,
    yolo: DetectionModel,
    device: str,
    max_det: int = 20,
    seq_len: int = SEQ_LEN,
    det_raw_window: int = 64,
    img_size: int = 640,
    nc: int = 2,
) -> torch.Tensor:
    """
    Build detection sequence (B, seq_len, max_det*5) by:
    1. Computing 64 raw ticks for each batch item
    2. Loading/caching detection vector for each tick
    3. Linspace 64 → seq_len

    Zero-pads for ticks before round start.
    """
    B = len(batch['game_id'])
    indices = list(np.linspace(0, det_raw_window - 1, seq_len, dtype=int))

    all_det_seqs = []
    for b in range(B):
        current_tick = int(batch['tick'][b].item())
        current_tick = current_tick - current_tick % 4
        demo_path = batch['game_id'][b]

        # Build raw 64 ticks (going back 4 ticks each step)
        ticks = [current_tick - (det_raw_window - 1 - j) * 4 for j in range(det_raw_window)]
        sampled_ticks = [ticks[k] for k in indices]  # 16 ticks

        det_seq = []
        for tick in sampled_ticks:
            if tick < 0:
                det_seq.append(torch.zeros(max_det * 5, device=device))
                continue
            frame_path = os.path.join(demo_path, f'tick_{max(0, tick)}.jpg')
            if not os.path.exists(frame_path):
                det_seq.append(torch.zeros(max_det * 5, device=device))
            else:
                det_seq.append(get_det_vector(frame_path, yolo, device,
                                              max_det, img_size, nc))

        all_det_seqs.append(torch.stack(det_seq).unsqueeze(0))  # (1, 16, max_det*5)

    return torch.cat(all_det_seqs)  # (B, 16, max_det*5)


def build_scene_embeddings(
    batch: Dict,
    yolo: DetectionModel,
    device: str,
    img_size: int = 640,
) -> torch.Tensor:
    """
    Build scene embeddings (B, seq_len, 512) from P3 + P5 backbone features.

    scene_frame_paths in batch is already linspaced to seq_len in DatasetIntent.
    P3/P5 features are extracted without grad (backbone is frozen).
    embeds_* heads run with grad for end-to-end scene embedding training.
    """
    B = len(batch['scene_frame_paths'])
    seq_len = len(batch['scene_frame_paths'][0])

    all_embeds = []
    for t in range(seq_len):
        frame_paths_t = [batch['scene_frame_paths'][b][t] for b in range(B)]

        # Extract P3 + P5 features (no grad, backbone frozen)
        p3, p5 = get_backbone_features_batch(frame_paths_t, yolo, device, img_size)

        # Run multi-scale embed head with grad (trainable)
        embed = yolo.embed_from_features(p3, p5)   # (B, 512)
        all_embeds.append(embed)

    return torch.stack(all_embeds, dim=1)  # (B, seq_len, 512)


def build_radar_embeddings(
    batch: Dict,
    radar_encoder: nn.Module,
    device: str,
) -> torch.Tensor:
    """Build radar embeddings (B, 16, 512) from pre-loaded radar frames."""
    # batch['radar_seq']: (B, 16, H, W, 3)
    radar_seq = batch['radar_seq'].to(device).permute(0, 1, 4, 2, 3)  # (B, 16, 3, H, W)
    B, T, C, H, W = radar_seq.shape
    flat = radar_seq.view(B * T, C, H, W)
    embeds = radar_encoder(flat)    # (B*T, 512)
    return embeds.view(B, T, 512)   # (B, 16, 512)


# ============================================================================
# Evaluate
# ============================================================================

def evaluate(
    val_loader: DataLoader,
    radar_encoder: nn.Module,
    yolo: DetectionModel,
    temporal_model: nn.Module,
    flow_head: nn.Module,
    audio_encoder: Optional[nn.Module],
    bce_loss: nn.Module,
    device: str,
    max_det: int,
    w_keys: float,
    metrics_file: str,
    use_audio: bool,
    epoch: int,
    log_dir: str,
    img_size: int = 640,
    nc: int = 2,
):
    radar_encoder.eval()
    yolo.eval()
    temporal_model.eval()
    flow_head.eval()
    if audio_encoder:
        audio_encoder.eval()

    total_loss = total_mse = total_flow = 0.0
    all_preds_keys, all_labels_keys = [], []
    mse_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='  Validation', leave=False):
            if batch is None:
                continue

            # Scene embeddings
            scene_embeds = build_scene_embeddings(batch, yolo, device, img_size)

            # Radar embeddings
            radar_embeds = build_radar_embeddings(batch, radar_encoder, device)

            # Detection sequence
            detect_embeds = build_detection_sequence(
                batch, yolo, device, max_det=max_det, nc=nc, img_size=img_size
            )  # (B, 16, 100)

            # Actions (B, 64, 22) — compressed 64→16 inside TemporalCrossTransformer
            actions_mouse = batch['actions_mouse'].to(device)  # (B, 64, 2)
            actions_keys = batch['actions_keys'].to(device)    # (B, 64, 20)
            action_seq = torch.cat([actions_mouse, actions_keys], dim=-1)  # (B, 64, 22)

            state_vec = batch['state_vec'].to(device)
            gt_mouse = batch['target_mouse'].to(device)
            gt_keys = batch['intent'].to(device)

            # Audio
            audio_embeds = None
            if use_audio and audio_encoder and 'audio_waveform' in batch:
                waveform = batch['audio_waveform'].to(device)
                raw_audio = audio_encoder(waveform)  # (B, 32, 512)
                # Linspace 32 → 16
                audio_idx = list(np.linspace(0, 31, SEQ_LEN, dtype=int))
                audio_embeds = raw_audio[:, audio_idx, :]  # (B, 16, 512)

            # Forward
            mouse_embed, policy_keys, _ = temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=action_seq,
                state_vec=state_vec,
                audio_seq=audio_embeds,
            )

            policy_mouse = flow_head.sample(mouse_embed)
            loss_mouse_mse = mse_fn(policy_mouse, gt_mouse)
            loss_flow = flow_head.compute_loss(mouse_embed, gt_mouse)
            loss_keys = bce_loss(policy_keys, gt_keys)

            total_loss += (loss_flow + w_keys * loss_keys).item()
            total_mse += loss_mouse_mse.item()
            total_flow += loss_flow.item()

            preds_keys = (torch.sigmoid(policy_keys) > 0.5).cpu().numpy()
            all_preds_keys.append(preds_keys)
            all_labels_keys.append(gt_keys.cpu().numpy())

    n = max(len(val_loader), 1)
    avg_loss = total_loss / n
    avg_mse = total_mse / n
    avg_flow = total_flow / n

    all_preds = np.concatenate(all_preds_keys, axis=0)
    all_labels = np.concatenate(all_labels_keys, axis=0)

    if SKLEARN_AVAILABLE:
        prec = precision_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
        rec = recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    else:
        tp = np.sum((all_labels == 1) & (all_preds == 1))
        fp = np.sum((all_labels == 0) & (all_preds == 1))
        fn = np.sum((all_labels == 1) & (all_preds == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    log_text = (f"Val — Loss: {avg_loss:.4f} | Flow: {avg_flow:.4f} | "
                f"MSE: {avg_mse:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    log_metrics(log_text, metrics_file)
    print(log_text)

    per_class = compute_per_class_metrics(all_labels, all_preds)
    per_class_file = os.path.join(log_dir, f'per_class_val_epoch_{epoch+1}.json')
    with open(per_class_file, 'w') as f:
        json.dump(per_class, f, indent=2)

    return avg_loss


# ============================================================================
# Training
# ============================================================================

def train():
    # ======== CONFIG ========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUN_NAME = datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S')
    CKPT_ROOT = './checkpoints_final'

    # Paths — update to your environment
    TRAIN_DATASET_JSON = '/mnt/ml/msirotkin/shock2/train_dataset.json'
    VAL_DATASET_JSON = '/mnt/ml/msirotkin/shock2/val_dataset.json'
    AUDIO_DIR = '/mnt/ml/msirotkin/shock2/AudioData'
    PRETRAINED_YOLO = None    # e.g. '/path/to/yolo11l_best.pt' from detect/train_yolo.py
    RESUME_CKPT = None        # e.g. '/path/to/epoch_5.pth' to resume training

    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    MAX_DET = 20
    LR = 1e-4
    W_KEYS = 100
    SEED = 42
    IMG_SIZE = 640
    NC = 2            # detection classes (body, head)
    USE_AUDIO = True
    NOISE_SCALE = 0.3
    FLOW_STEPS = 5

    CKPT_DIR = os.path.join(CKPT_ROOT, RUN_NAME, 'checkpoints')
    LOG_DIR = os.path.join(CKPT_ROOT, RUN_NAME, 'logs')
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    METRICS_FILE = os.path.join(LOG_DIR, 'metrics.txt')

    seed_everything(SEED)

    # ======== DATASETS ========
    train_dataset = CSRoundDataset(
        TRAIN_DATASET_JSON,
        use_audio=USE_AUDIO,
        audio_dir=AUDIO_DIR,
        radar_window=32,
        scene_window=64,
        actions_window=64,
    )
    val_dataset = CSRoundDataset(
        VAL_DATASET_JSON,
        use_audio=USE_AUDIO,
        audio_dir=AUDIO_DIR,
        radar_window=32,
        scene_window=64,
        actions_window=64,
    )

    # Fixed-size outputs → simple collate (no ExactMatchBucketSampler needed)
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        result = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([b[key] for b in batch])
            elif isinstance(batch[0][key], list):
                result[key] = [b[key] for b in batch]
            else:
                result[key] = [b[key] for b in batch]
        return result

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ======== MODELS ========
    checkpoint = None
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        checkpoint = torch.load(RESUME_CKPT, map_location='cpu')
        print(f'Loaded checkpoint from {RESUME_CKPT}')

    # Radar encoder
    radar_encoder = RadarEncoderEffB0(pretrained=True, in_ch=3, out_ch=64, embed_dim=512).to(device)
    if checkpoint and 'radar_encoder' in checkpoint:
        radar_encoder.load_state_dict(checkpoint['radar_encoder'], strict=False)
        print('RadarEncoder: loaded from checkpoint')

    # YOLO (l-scale, frozen backbone, trainable embeds)
    yaml_path = os.path.join(os.path.dirname(__file__), 'yolo11.yaml')
    yolo = DetectionModel(cfg=yaml_path, nc=NC, scale='l').to(device)
    if PRETRAINED_YOLO and os.path.exists(PRETRAINED_YOLO):
        load_pretrained_weights(yolo, PRETRAINED_YOLO)
        print(f'YOLO: loaded pretrained from {PRETRAINED_YOLO}')
    if checkpoint and 'yolo' in checkpoint:
        yolo.load_state_dict(checkpoint['yolo'], strict=False)
        print('YOLO: loaded from checkpoint')

    # Freeze backbone, keep embeds trainable
    for name, p in yolo.named_parameters():
        p.requires_grad_(name.startswith('embeds'))

    embed_params = sum(p.numel() for p in yolo.parameters() if p.requires_grad)
    total_yolo = sum(p.numel() for p in yolo.parameters())
    print(f'YOLO total: {total_yolo:,}  |  embeds (trainable): {embed_params:,}')

    # Audio encoder
    audio_encoder = None
    if USE_AUDIO:
        audio_encoder = AudioEncoder(embed_dim=512, target_embeddings=32).to(device)
        if checkpoint and 'audio_encoder' in checkpoint:
            audio_encoder.load_state_dict(checkpoint['audio_encoder'], strict=False)
            print('AudioEncoder: loaded from checkpoint')
        else:
            print('AudioEncoder: initialized from scratch')
        n_audio = sum(p.numel() for p in audio_encoder.parameters())
        print(f'AudioEncoder params: {n_audio:,}')

    # Temporal model (d_model=384, SEQ_LEN=16)
    temporal_model = TemporalCrossTransformer(
        use_audio=USE_AUDIO,
        radar_dim=512,
        scene_dim=512,
        detection_dim=MAX_DET * 5,  # 100
        actions_dim=22,
        state_dim=95,
        audio_dim=512,
        seq_len=SEQ_LEN,
        scene_seq_len=SCENE_SEQ_LEN,
        actions_seq_len=ACTION_SEQ_LEN,
        d_model=384,
        num_heads=8,
        depth=6,
    ).to(device)
    if checkpoint and 'temporal_model' in checkpoint:
        state = checkpoint['temporal_model']
        # Remove DataParallel prefix if present
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        temporal_model.load_state_dict(state, strict=False)
        print('TemporalModel: loaded from checkpoint')
    n_temporal = sum(p.numel() for p in temporal_model.parameters())
    print(f'TemporalTransformer params: {n_temporal:,}')

    # Flow action head (context_dim MUST = d_model = 384)
    flow_head = FlowActionHead(
        context_dim=384,
        action_dim=2,
        hidden_dim=256,
        noise_scale=NOISE_SCALE,
        num_steps=FLOW_STEPS,
    ).to(device)
    if checkpoint and 'flow_head' in checkpoint:
        flow_head.load_state_dict(checkpoint['flow_head'], strict=False)
        print('FlowActionHead: loaded from checkpoint')
    n_flow = sum(p.numel() for p in flow_head.parameters())
    print(f'FlowActionHead params: {n_flow:,}')

    # ======== OPTIMIZER ========
    bce_loss = nn.BCEWithLogitsLoss()

    params = (
        list(radar_encoder.parameters()) +
        [p for p in yolo.parameters() if p.requires_grad] +  # embeds only
        list(temporal_model.parameters()) +
        list(flow_head.parameters())
    )
    if audio_encoder:
        params += list(audio_encoder.parameters())

    optimizer = optim.AdamW(params, lr=LR)
    if checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint.get('epoch', 0) if checkpoint else 0
    best_val_loss = float('inf')

    # ======== TRAINING LOOP ========
    for epoch in range(start_epoch, NUM_EPOCHS):
        radar_encoder.train()
        yolo.eval()           # backbone eval (frozen), embeds head trains via build_scene_embeddings
        temporal_model.train()
        flow_head.train()
        if audio_encoder:
            audio_encoder.train()

        total_loss = total_flow_loss = 0.0
        all_preds_train, all_labels_train = [], []
        iter_preds, iter_labels = [], []

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')):
            if batch is None:
                continue
            torch.cuda.empty_cache()

            # Scene embeddings
            scene_embeds = build_scene_embeddings(batch, yolo, device, IMG_SIZE)

            # Radar embeddings
            radar_embeds = build_radar_embeddings(batch, radar_encoder, device)

            # Detection sequence
            detect_embeds = build_detection_sequence(
                batch, yolo, device,
                max_det=MAX_DET, nc=NC, img_size=IMG_SIZE,
            )  # (B, 16, 100)

            # Actions (B, 64, 22) from DatasetIntent — compressed 64→16 in TemporalCrossTransformer
            actions_mouse = batch['actions_mouse'].to(device)
            actions_keys = batch['actions_keys'].to(device)
            action_seq = torch.cat([actions_mouse, actions_keys], dim=-1)  # (B, 64, 22)

            state_vec = batch['state_vec'].to(device)
            gt_mouse = batch['target_mouse'].to(device)
            gt_keys = batch['intent'].to(device)

            # Audio (linspace 32 → 16)
            audio_embeds = None
            if USE_AUDIO and audio_encoder and 'audio_waveform' in batch:
                waveform = batch['audio_waveform'].to(device)
                raw_audio = audio_encoder(waveform)  # (B, 32, 512)
                audio_idx = list(np.linspace(0, 31, SEQ_LEN, dtype=int))
                audio_embeds = raw_audio[:, audio_idx, :]  # (B, 16, 512)

            # Forward
            mouse_embed, policy_keys, _ = temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=action_seq,
                state_vec=state_vec,
                audio_seq=audio_embeds,
            )

            loss_mouse = flow_head.compute_loss(mouse_embed, gt_mouse)
            loss_keys = bce_loss(policy_keys, gt_keys)
            loss = loss_mouse + W_KEYS * loss_keys

            # Debug every 100 iters
            if i > 0 and i % 100 == 0:
                with torch.no_grad():
                    sampled = flow_head.sample(mouse_embed)
                print(f'\n[iter {i}] flow={loss_mouse.item():.4f} keys={loss_keys.item():.4f}')
                print(f'[iter {i}] gt_mouse mean={gt_mouse.mean():.3f} std={gt_mouse.std():.3f}')
                print(f'[iter {i}] pred_mouse mean={sampled.mean():.3f} std={sampled.std():.3f}')
                if iter_preds:
                    iter_arr_p = np.concatenate(iter_preds, axis=0)
                    iter_arr_l = np.concatenate(iter_labels, axis=0)
                    iter_metrics = compute_per_class_metrics(iter_arr_l, iter_arr_p)
                    worst = sorted(
                        [(k, m) for k, m in iter_metrics.items() if m['support'] > 5],
                        key=lambda x: x[1]['recall']
                    )
                    print(f'[iter {i}] Worst keys by recall:')
                    for kn, m in worst[:3]:
                        print(f'  {kn:>8}: recall={m["recall"]:.3f} supp={m["support"]}')
                    iter_preds, iter_labels = [], []

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_flow_loss += loss_mouse.item()

            with torch.no_grad():
                preds_k = (torch.sigmoid(policy_keys) > 0.5).cpu().numpy()
                labels_k = gt_keys.cpu().numpy()
                all_preds_train.append(preds_k)
                all_labels_train.append(labels_k)
                iter_preds.append(preds_k)
                iter_labels.append(labels_k)

        # Train metrics
        n = max(len(train_loader), 1)
        avg_train = total_loss / n
        avg_flow = total_flow_loss / n
        all_p = np.concatenate(all_preds_train, axis=0)
        all_l = np.concatenate(all_labels_train, axis=0)
        if SKLEARN_AVAILABLE:
            trn_prec = precision_score(all_l.flatten(), all_p.flatten(), zero_division=0)
            trn_rec = recall_score(all_l.flatten(), all_p.flatten(), zero_division=0)
        else:
            tp = np.sum((all_l == 1) & (all_p == 1))
            fp = np.sum((all_l == 0) & (all_p == 1))
            fn = np.sum((all_l == 1) & (all_p == 0))
            trn_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            trn_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        trn_log = (f'Train — Loss: {avg_train:.4f} | Flow: {avg_flow:.4f} | '
                   f'Prec: {trn_prec:.4f} | Rec: {trn_rec:.4f}')
        log_metrics(trn_log, METRICS_FILE)
        print(trn_log)

        # Validation
        avg_val = evaluate(
            val_loader, radar_encoder, yolo, temporal_model, flow_head,
            audio_encoder, bce_loss, device,
            MAX_DET, W_KEYS, METRICS_FILE, USE_AUDIO, epoch, LOG_DIR,
            img_size=IMG_SIZE, nc=NC,
        )

        epoch_log = f'Epoch [{epoch+1}/{NUM_EPOCHS}] Train: {avg_train:.4f} | Val: {avg_val:.4f}'
        log_metrics(epoch_log, METRICS_FILE)
        print(epoch_log)

        # Save checkpoint
        ckpt_data = {
            'epoch': epoch + 1,
            'radar_encoder': radar_encoder.state_dict(),
            'yolo': yolo.state_dict(),
            'temporal_model': temporal_model.state_dict(),
            'flow_head': flow_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': avg_train,
            'val_loss': avg_val,
        }
        if audio_encoder:
            ckpt_data['audio_encoder'] = audio_encoder.state_dict()

        torch.save(ckpt_data, os.path.join(CKPT_DIR, f'epoch_{epoch+1}.pth'))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt_data, os.path.join(CKPT_DIR, 'best.pth'))
        print(f'Saved checkpoint (val={avg_val:.4f}, best={best_val_loss:.4f})')


if __name__ == '__main__':
    train()
