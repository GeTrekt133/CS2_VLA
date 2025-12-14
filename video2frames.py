import cv2
import os
from tqdm import tqdm
from demoparser2 import DemoParser
from pos_matching import radar_pos_matching
from demo_parser2 import parse_ticks_csv


output_main_dir = r"D:\FramesDataset"
cs_dir = r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo'
record_demo_dir = r'D:\RecordDemos'
bomb_orange = cv2.imread('bomb_dropped_crop.jpg')
bomb_yellow = cv2.imread('orange_to_yellow.png')
bomb_violet = cv2.imread('orange_to_violet.png')
bomb_red = cv2.imread('orange_to_red.png')
bomb_white = cv2.imread('orange_to_white.png')
bomb_green = cv2.imread('orange_to_green.png')
bomb_blue = cv2.imread('orange_to_cyan.png')
img = cv2.imread('vanila_radar.jpg')
img = img[8:425, 10:574]
target_steam = 76561198386265483
size = 15
fov = 70
base_width = 20
length = 40
alpha_max = 1
right_up = (1850, 1700)
left_bot = (-3200, -3300)
coefX = (abs(right_up[0]) + abs(left_bot[0])) / img.shape[1]
coefY = (abs(right_up[1]) + abs(left_bot[1])) / img.shape[0]
team_colors = [
    (211, 0, 148),   # фиолетовый
    (0, 255, 255),   # жёлтый
    (0, 100, 220),   # оранжевый
    (0, 255, 0),     # зелёный
    (255, 0, 0),    # синий
]

for demo in tqdm(os.listdir(record_demo_dir)):
    demo_path = os.path.join(cs_dir, f'{demo}.dem')
    parser = DemoParser(demo_path)
    df = parse_ticks_csv(demo_path)
    tmp_df = df[df['tick'] == 1000]
    target_team = tmp_df[tmp_df['steamid'] == target_steam]['team_num'].item()
    teammates = list(tmp_df[tmp_df['team_num'] == target_team]['user_id'])
    teammates_steamid = list(map(int, tmp_df[tmp_df['team_num'] == target_team]['steamid']))
    start_df = parser.parse_event('round_prestart')
    round_start_ticks = list(start_df['tick'][3:])
    demo_dir = os.path.join(record_demo_dir, demo)
    print(round_start_ticks)
    for i, round in enumerate(os.listdir(demo_dir)):
        cd_dead = {}
        for user_id in tmp_df['user_id']:
            cd_dead[user_id] = [10, True]
        dict_colors = {}
        for j, teammate in enumerate(teammates):
            dict_colors[teammate] = team_colors[j]
        dict_colors[-1] = (0, 0, 255)
        output_dir = os.path.join(output_main_dir, demo)
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(demo_dir, round)
        start_tick = round_start_ticks[i] - round_start_ticks[i] % 4
        print(start_tick, round_start_ticks[i], i)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎞 FPS: {fps:.2f}, кадров всего: {total_frames}")
        frame_idx = -28
        with tqdm(total=total_frames, desc="Saving frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                tick = start_tick if frame_idx < 0 else frame_idx * 4 + start_tick
                radar, cd_dead = radar_pos_matching(tick, df, img, target_steam, cd_dead, dict_colors, teammates_steamid, teammates, size, fov, base_width, length, alpha_max, left_bot, coefX, coefY, bomb_orange, bomb_yellow, bomb_violet, bomb_red, bomb_white, bomb_green, bomb_blue)
                radar = cv2.resize(radar, (424, 314))
                frame[6:320, 7:431] = radar
                resized_frame = cv2.resize(frame, (640, 640))
                frame_path = os.path.join(output_dir, f"tick_{tick}.jpg")
                cv2.imwrite(frame_path, resized_frame)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"✅ Сохранено кадров: {frame_idx}")
        print(f"📂 Все кадры сохранены в: {os.path.abspath(output_dir)}")
