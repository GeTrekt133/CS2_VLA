import time, os
import cv2
import numpy as np
import pyautogui
import torch
import torchvision.ops as ops
from tqdm import tqdm
from ultralytics import YOLO
from src.Yolo import DetectionModel, load_pretrained_weights, Detect


# MODEL_PATH = "yolov8_csgo_cs2_model.pt"
# MODEL_PATH = "yolo11x-pose.pt"


# detection_model = YOLO(MODEL_PATH).to('cuda')
yolo = DetectionModel(cfg=r"C:\Users\misas\CS2_NN\src\yolo11n.yaml")
detection_model = load_pretrained_weights(yolo, "yolo11n.pt").to('cuda').eval()
save_root = r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1-bboxes"
root = r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1"
for frame in tqdm(os.listdir(root)):
    image_path = os.path.join(root, frame)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, (640, 640))  # YOLO input size
    img = img.astype(np.float32) / 255.0  # нормализация 0-1
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to('cuda')
    t = time.time()
    with torch.no_grad():
        output = detection_model(img_tensor)
    bboxes = Detect.postprocess(output[0], 100, 80)
    img_show = cv2.imread(image_path)
    img_show = cv2.resize(img_show, (640, 640))
    boxes_xyxy, conf, cls = bboxes[0][:, :4], bboxes[0][:, 4], bboxes[0][:, 5]
    boxes_xyxy = ops.box_convert(boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy")
    keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.45)
    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]
    for (x1, y1, x2, y2), c, cl in zip(boxes_xyxy.cpu(), conf.cpu(), cls.cpu()):
        label = f"{int(cl)} {c:.2f}"
        if cl == 0:
            cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_root, frame), img_show)
# print(detection_results)