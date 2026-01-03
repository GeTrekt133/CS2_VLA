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
from Dataloader import ExactMatchBucketSampler
# from Dataset import CSRoundDataset
from DatasetIntent import CSRoundDataset
from RadarEncoder import RadarEncoderEffB0
from Yolo import DetectionModel, Detect, load_pretrained_weights
from TemporalTransformer import TemporalCrossTransformer
from sklearn.metrics import precision_score, recall_score


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_metrics(text, save_file):
    with open(save_file, "a") as f:
        f.write(text + "\n")

def detections(batch, model, max_det, batch_size, device):
    detects_stack = []
    # print(batch["tick"], batch["scene_seq"].shape, batch["radar_seq"].shape)
    for i in range(batch_size):
        tick = batch['tick'][i].item() - batch['tick'][i].item() % 4
        # print(f"/mnt/ml/msirotkin/shock2/FramesDataset/{batch['game_id'][i]}/tick_{tick}.jpg")
        img = cv2.imread(f"/mnt/ml/msirotkin/shock2/FramesDataset/{batch['game_id'][i]}/tick_{tick}.jpg")
        # print(f"D:\\FramesDataset\\{batch['game_id'][0]}\\tick_{batch['tick'].item()}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # нормализация 0-1
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            detects = model(img)[0][0]
        bboxes = Detect.postprocess(detects, 100, 80)
        boxes_xyxy, conf, cls = bboxes[0][:, :4], bboxes[0][:, 4], bboxes[0][:, 5]
        boxes_xyxy = ops.box_convert(boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy")
        keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.3)
        boxes_xyxy = boxes_xyxy[keep]
        conf = conf[keep]
        cls = cls[keep]
        stack = []
        for i, cl in enumerate(cls):
            if cl == 0:
                stack.append(torch.cat([boxes_xyxy[i] / 640, conf[i].unsqueeze(0)]))
        if len(stack) == 0:
            detects = torch.zeros(max_det * 5, device=device)
        else:
            detects = torch.cat(stack)
            n = detects.shape[0]
            pad_len = max_det * 5 - n
            if n > max_det * 5:
                detects = detects[:max_det * 5]
                pad_len = 0
            detects = F.pad(detects, (0, pad_len))

        # del img, detects, bboxes, boxes_xyxy, conf, cls, stack
        detect_embeds = detects.view((1, 1, max_det * 5))
        detects_stack.append(detect_embeds)
    dets = torch.cat(detects_stack)
    return dets

def evaluate(val_loader, radar_encoder, yolo, temporal_model, mse_loss, bce_loss, device, max_det, batch_size, w_keys, metrics_file):
    radar_encoder.eval()
    yolo.eval()
    temporal_model.eval()

    total_loss = 0.0
    total_mse_mouse = 0.0

    all_preds_keys = []
    all_labels_keys = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            try:
                torch.cuda.empty_cache()
                scene_seq = batch["scene_seq"].to(device).permute(0,1,4,2,3)
                radar_seq = batch["radar_seq"].to(device).permute(0,1,4,2,3)
                actions_mouse = batch["actions_mouse"].to(device)
                actions_keys = batch["actions_keys"].to(device)
                state_vec = batch["state_vec"].to(device)
                gt_mouse = batch["target_mouse"].to(device)
                gt_keys = batch["intent"].to(device)
                detect_embeds = detections(batch, yolo, max_det, batch_size, device)
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
                policy_mouse, policy_keys, _ = temporal_model(
                    radar_seq=radar_embeds,
                    scene_seq=scene_embeds,
                    detection_seq=detect_embeds,
                    action_seq=torch.cat([actions_mouse, actions_keys], dim=-1),
                    state_vec=state_vec
                )

                # === Loss ===
                loss_mouse = mse_loss(policy_mouse, gt_mouse)
                loss_keys = bce_loss(policy_keys, gt_keys)
                total_loss += (loss_mouse + w_keys * loss_keys).item()
                total_mse_mouse += loss_mouse.item()

                # === Collect predictions for metrics ===
                preds_keys = torch.sigmoid(policy_keys).detach().cpu().numpy() > 0.5
                labels_keys = gt_keys.detach().cpu().numpy()

                all_preds_keys.append(preds_keys)
                all_labels_keys.append(labels_keys)
                # print(preds_keys, labels_keys, loss_mouse)
            except:
                print("val error")

    # === Aggregate metrics ===
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
    device = "cuda"

    RUN_NAME = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    CKPT_ROOT = "./checkpoints2"
    TRAIN_DATASET_JSON = "/mnt/ml/msirotkin/shock2/train_dataset.json"
    VAL_DATASET_JSON = "/mnt/ml/msirotkin/shock2/val_dataset.json"
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    MAX_DET = 20
    LR = 1e-4
    W_KEYS = 1
    CKPT_DIR = os.path.join(CKPT_ROOT, RUN_NAME, "checkpoints")
    LOG_DIR = os.path.join(CKPT_ROOT, RUN_NAME, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    METRICS_FILE = os.path.join(LOG_DIR, "metrics.txt")
    SEED = 42
    seed_everything(SEED)

    train_dataset = CSRoundDataset(TRAIN_DATASET_JSON)
    val_dataset = CSRoundDataset(VAL_DATASET_JSON)

    train_sampler = ExactMatchBucketSampler(CSRoundDataset(TRAIN_DATASET_JSON, sampler=True), batch_size=BATCH_SIZE, shuffle=True)
    val_sampler = ExactMatchBucketSampler(CSRoundDataset(VAL_DATASET_JSON, sampler=True), batch_size=BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)

    # checkpoint = torch.load("/mnt/ml/msirotkin/shock2/checkpoints/best_epoch_16.pth", map_location="cuda")
    radar_encoder = RadarEncoderEffB0(pretrained=True, in_ch=3, out_ch=64, embed_dim=512).to(device)
    # radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=False)
    yolo = DetectionModel(cfg="/mnt/ml/msirotkin/shock2/src/yolo11n.yaml").to(device) #TODO сделать заморозку слоев детектора, оставить эмбеддер + написать логику прокидывания детекций
    # yolo = load_pretrained_weights(yolo, "/mnt/ml/msirotkin/shock2/yolo11n.pt").to(device).eval()
    # yolo.load_state_dict(checkpoint["yolo"], strict=False)
    temporal_model = TemporalCrossTransformer().to(device)
    # temporal_model.load_state_dict(checkpoint["temporal_model"], strict=False)

    # radar_encoder = nn.DataParallel(radar_encoder)
    # yolo = nn.DataParallel(yolo)
    temporal_model = nn.DataParallel(temporal_model)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    params = list(radar_encoder.parameters()) + list(yolo.parameters()) + list(temporal_model.parameters())
    optimizer = optim.AdamW(params, lr=LR)
    # optimizer.load_state_dict(checkpoint["optimizer"])

    best_val_loss = float("inf")

    for epoch in range(0, NUM_EPOCHS):
        radar_encoder.train()
        yolo.eval()
        for name, param in yolo.named_parameters():
            if "embeds" in name:
                param.requires_grad = True
                # print('embed')
            elif "model" in name:
                param.requires_grad = False
        temporal_model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            # try:
            torch.cuda.empty_cache()
            scene_seq = batch["scene_seq"].to(device).permute(0,1,4,2,3)
            radar_seq = batch["radar_seq"].to(device).permute(0,1,4,2,3)
            actions_mouse = batch["actions_mouse"].to(device)
            actions_keys = batch["actions_keys"].to(device)
            state_vec = batch["state_vec"].to(device)
            gt_mouse = batch["target_mouse"].to(device)
            gt_keys = batch["intent"].to(device)
            detect_embeds = detections(batch, yolo, MAX_DET, BATCH_SIZE, device)
            # print(scene_seq.shape, radar_seq.shape)
            radar_embeds = torch.stack(
                [radar_encoder(radar_seq[:, t]) for t in range(radar_seq.shape[1])],
                dim=1
            )
            scene_embeds = torch.stack(
                [yolo(scene_seq[:, t])[1] for t in range(scene_seq.shape[1])],
                dim=1
            )
            policy_mouse, policy_keys, _ = temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=torch.cat([actions_mouse, actions_keys], dim=-1),
                state_vec=state_vec
            )
            # print(policy_mouse.shape, policy_keys.shape, gt_mouse.shape, gt_keys.shape)

            loss_mouse = mse_loss(policy_mouse, gt_mouse)
            loss_keys = bce_loss(policy_keys, gt_keys)
            loss = loss_mouse + W_KEYS * loss_keys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # except:
            #     print("train error")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(val_loader, radar_encoder, yolo, temporal_model, mse_loss, bce_loss, device, MAX_DET, BATCH_SIZE, W_KEYS, METRICS_FILE)

        log_text = f"Epoch [{epoch+1}/{NUM_EPOCHS}] — Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        log_metrics(log_text, METRICS_FILE)
        print(log_text)

        # ====== SAVE CHECKPOINT ======
        # if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "epoch": epoch + 1,
            "radar_encoder": radar_encoder.state_dict(),
            "yolo": yolo.state_dict(),
            "temporal_model": temporal_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth"))

        print(f"✅ Saved new best model with val_loss={avg_val_loss:.4f}")


if __name__ == "__main__":
    train()
