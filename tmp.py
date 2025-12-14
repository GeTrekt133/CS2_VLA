import cv2
import os
from tqdm import tqdm

video_path = r"C:\Users\misas\Videos\2025-10-14 17-49-29.mkv"  # путь к твоему видео
output_dir = "frames2"  # папка для кадров

# Создаём папку, если её нет
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞 FPS: {fps:.2f}, кадров всего: {total_frames}")

frame_idx = -30
with tqdm(total=total_frames, desc="Saving frames", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_idx % 300 == 0 and flag == True and frame_idx != 0:
        #     frame_idx -= 1
        #     flag = False
        # else:
        #     flag = True
        tick = 7683 if frame_idx < 0 else frame_idx * 4 + 7683
        frame_path = os.path.join(output_dir, f"tick_{tick}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_idx += 1
        pbar.update(1)

cap.release()
print(f"✅ Сохранено кадров: {frame_idx}")
print(f"📂 Все кадры сохранены в: {os.path.abspath(output_dir)}")
