import cv2
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from demo_parser2 import parse_ticks_csv


def angle_in_cone(angle, left, right):
    a = (angle - left) % (2*math.pi)
    b = (right - left) % (2*math.pi)
    return a <= b

def radar_pos_matching(tick, df, default_img, target_steam, cd_dead, dict_colors, teammates_steamid, teammates, size, fov, base_width, length, alpha_max, left_bot, coefX, coefY, bomb_orange, bomb_yellow, bomb_violet, bomb_red, bomb_white, bomb_green, bomb_blue):
    img = default_img.copy()
    tick_df = df[df['tick'] == tick].sort_values('user_id')
    row_target = tick_df[tick_df['steamid'] == target_steam]
    tick_df = tick_df[tick_df['steamid'] != target_steam]
    tick_df = pd.concat([tick_df, row_target]).reset_index(drop=True)
    for i in range(len(tick_df)):
        x, y = tick_df['X'].iloc[i], tick_df['Y'].iloc[i]
        x, y = round(x / coefX - left_bot[0] / coefX), img.shape[0] - round(y / coefY - left_bot[1] / coefY)
        user_id = tick_df['user_id'].iloc[i]
        steamid = tick_df['steamid'].iloc[i]
        if user_id not in teammates:
            user_id = -1
        if tick_df['is_alive'].iloc[i] == True and user_id != -1 or tick_df['spotted'].iloc[i] == True and user_id == -1 and tick_df['is_alive'].iloc[i] == True:
            cd_dead[tick_df['user_id'].iloc[i]] = [10, True]
            if tick_df['owner'].iloc[i] is not pd.NA and steamid != tick_df['owner'].iloc[i] or tick_df['bomb_dropped'].iloc[i] == True or tick_df['bomb_planted'].iloc[i] == True:
                img = cv2.circle(img, (x, y), 10, color=dict_colors[user_id], thickness=-1)
            elif tick_df['owner'].iloc[i] is not pd.NA and (tick_df['owner'].iloc[i] in teammates_steamid and round(tick_df['bombX'].iloc[i]) == round(tick_df['X'].iloc[i]) and \
                    round(tick_df['bombY'].iloc[i]) == round(tick_df['Y'].iloc[i]) or tick_df['spotted'].iloc[i] == True and \
                    tick_df['owner'].iloc[i] not in teammates_steamid) and tick_df['bomb_planted'].iloc[i] == False:
                match dict_colors[user_id]:
                    case (211, 0, 148): bomb_img = bomb_violet
                    case (0, 255, 255): bomb_img = bomb_yellow
                    case (0, 100, 220): bomb_img = bomb_orange
                    case (0, 255, 0): bomb_img = bomb_green
                    case (255, 0, 0): bomb_img = bomb_blue
                    case (0, 0, 255): bomb_img = bomb_red
                # img = cv2.circle(img, (x, y), 10, color=dict_colors[user_id], thickness=-1)
                bombX, bombY = tick_df['bombX'].iloc[i], tick_df['bombY'].iloc[i]
                bombX, bombY = round(bombX / coefX - left_bot[0] / coefX), img.shape[0] - round(bombY / coefY - left_bot[1] / coefY)
                img[bombY - 8 : bomb_img.shape[0] + bombY - 8, bombX - 12 : bomb_img.shape[1] + bombX - 12] = bomb_img
            if user_id != -1:
                yaw_rad = math.pi*2 - math.radians(tick_df['yaw'].iloc[i])
                dir_x = math.cos(yaw_rad)
                dir_y = math.sin(yaw_rad)
                tip_x, tip_y  = x + dir_x * size, y + dir_y * size
                angle_offset = math.radians(30)  # угол для основания
                left_x = x + math.cos(yaw_rad + angle_offset) * (base_width/2)
                left_y = y + math.sin(yaw_rad + angle_offset) * (base_width/2)
                right_x = x + math.cos(yaw_rad - angle_offset) * (base_width/2)
                right_y = y + math.sin(yaw_rad - angle_offset) * (base_width/2)

                triangle_cnt = np.array([
                    [int(tip_x), int(tip_y)],
                    [int(left_x), int(left_y)],
                    [int(right_x), int(right_y)]
                ])

                cv2.fillPoly(img, [triangle_cnt], color=(230,230,230))
                
                if steamid == target_steam:
                    left_bound = yaw_rad - math.radians(fov / 2)
                    right_bound = yaw_rad + math.radians(fov / 2)
                    h, w = img.shape[:2]
                    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

                    dx = X - x
                    dy = Y - y
                    distances = np.sqrt(dx**2 + dy**2)
                    angles = np.arctan2(dy, dx)
                    mask = (distances <= length) & angle_in_cone(angles, left_bound, right_bound)
                    alpha = np.zeros_like(distances, dtype=np.float32)
                    alpha[mask] = alpha_max * (1 - (distances[mask] / length))

                    overlay = np.zeros_like(img, dtype=np.uint8)
                    overlay[:, :] = (230, 230, 230)

                    for c in range(3):
                        img[:, :, c] = img[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
        
        if tick_df['is_alive'].iloc[i] == False and cd_dead[tick_df['user_id'].iloc[i]][0] != 0 and cd_dead[tick_df['user_id'].iloc[i]][1] == True:
            cd_dead[tick_df['user_id'].iloc[i]][0] -= 1
            color = dict_colors[user_id]
            cv2.line(img, (x - 8, y - 8), (x + 8, y + 8), color, 5)
            cv2.line(img, (x - 8, y + 8), (x + 8, y - 8), color, 5)
        elif cd_dead[tick_df['user_id'].iloc[i]][0] == 0:
            cd_dead[tick_df['user_id'].iloc[i]][1] = False

        if tick_df['owner'].iloc[i] is not pd.NA and tick_df['owner'].iloc[i] in teammates_steamid and (tick_df['bomb_dropped'].iloc[i] == True or tick_df['bomb_planted'].iloc[i] == True):
            bomb_img = bomb_white
            bombX, bombY = tick_df['bombX'].iloc[i], tick_df['bombY'].iloc[i]
            bombX, bombY = round(bombX / coefX - left_bot[0] / coefX), img.shape[0] - round(bombY / coefY - left_bot[1] / coefY)
            img[bombY - 8 : bomb_img.shape[0] + bombY - 8, bombX - 12 : bomb_img.shape[1] + bombX - 12] = bomb_img
    
    return img, cd_dead


if __name__ == "__main__":
    target_steam = 76561198386265483
    size = 15
    fov = 70
    base_width = 20
    length = 40
    alpha_max = 1
    right_up = (1850, 1700)
    left_bot = (-3200, -3300)
    df = parse_ticks_csv(r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\1-469477d6-c6da-4c50-bd56-bd7b54426192-1-1.dem")
    # df.to_csv('df.csv', index=False)
    # df = pd.read_csv('df.csv')

    bomb_orange = cv2.imread('bomb_dropped_crop.jpg')
    bomb_yellow = cv2.imread('orange_to_yellow.png')
    bomb_violet = cv2.imread('orange_to_violet.png')
    bomb_red = cv2.imread('orange_to_red.png')
    bomb_white = cv2.imread('orange_to_white.png')
    bomb_green = cv2.imread('orange_to_green.png')
    bomb_blue = cv2.imread('orange_to_cyan.png')
    img = cv2.imread('vanila_radar.jpg')
    img = img[8:425, 10:574]
    # cv2.imwrite('crop_radar.jpg', img)
    tmp_df = df[df['tick'] == 1000]
    target_team = tmp_df[tmp_df['steamid'] == target_steam]['team_num'].item()
    teammates = list(tmp_df[tmp_df['team_num'] == target_team]['user_id'])
    teammates_steamid = list(map(int, tmp_df[tmp_df['team_num'] == target_team]['steamid']))
    cd_dead = {}
    for user_id in tmp_df['user_id']:
        cd_dead[user_id] = [10, True]
    team_colors = [
            (211, 0, 148),   # фиолетовый
            (0, 255, 255),   # жёлтый
            (0, 100, 220),   # оранжевый
            (0, 255, 0),     # зелёный
            (255, 0, 0),    # синий
        ]
    dict_colors = {}
    for i, teammate in enumerate(teammates):
        dict_colors[teammate] = team_colors[i]
    dict_colors[-1] = (0, 0, 255)
    coefX = (abs(right_up[0]) + abs(left_bot[0])) / img.shape[1]
    coefY = (abs(right_up[1]) + abs(left_bot[1])) / img.shape[0]
    img = cv2.imread('vanila_radar.jpg')
    img = img[8:425, 10:574]
    for j in tqdm(range(0, len(df) // 10, 10)):
    # for j in tqdm(range(3000, 3200, 10)):
        radar = radar_pos_matching(j, df, img, target_steam, cd_dead, dict_colors, teammates_steamid, teammates, size, fov, base_width, length, alpha_max, left_bot, coefX, coefY, bomb_orange, bomb_yellow, bomb_violet, bomb_red, bomb_white, bomb_green, bomb_blue)
        path = f"C:\\Users\\misas\CS2_NN\\radar_dataset_mirage\\radar_tick_{j}.jpg"
        cv2.imwrite(path, radar)
