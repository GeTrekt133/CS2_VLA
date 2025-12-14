import cv2
import numpy as np

def world_to_local(vec, yaw, pitch):
    # CS2 convention:
    # yaw — вращение вокруг Z (горизонт)
    # pitch — вращение вокруг Y (вертикаль, вверх/вниз)
    yaw = np.radians(yaw)
    pitch = np.radians(-pitch)  # инвертируем pitch

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # камера смотрит вдоль +X в локальных координатах
    forward = np.array([cp * cy, cp * sy, sp])
    right   = np.array([-sy, cy, 0])
    up      = np.cross(right, forward)

    mat = np.vstack([right, up, forward]).T
    return mat.T @ vec


def project_to_screen(local, fov_deg, width, height):
    fov = np.radians(fov_deg)
    aspect = width / height
    fx = 1 / np.tan(fov / 2)
    fy = fx * aspect

    x, y, z = local
    # if z <= 0:
    #     return None

    ndc_x = (x / z) * fx
    ndc_y = (y / z) * fy

    screen_x = width  / 2 * (1 + ndc_x)
    screen_y = height / 2 * (1 + ndc_y)
    return np.array([screen_x, screen_y])

# ----------------------------
# Твои данные
pos_target = np.array([-1217.903, -1231.7535, -167.96875])
pos_obs    = np.array([-946.24567, -1429.6041, -131.375])
yaw, pitch = 2.475006, 143.83987
fov = 90
W, H = 640, 640

PLAYER_HEIGHT = 72
PLAYER_WIDTH = 32
# ----------------------------

# Загружаем кадр или делаем пустой
frame = cv2.imread(r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1\tick_34484.jpg")
if frame is None:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
else:
    frame = cv2.resize(frame, (W, H))

# Проецируем
pos_head = pos_target + np.array([0, 0, PLAYER_HEIGHT])
pos_feet = pos_target
left_pos  = pos_target + np.array([-PLAYER_WIDTH / 2, 0, PLAYER_HEIGHT / 2])
right_pos = pos_target + np.array([ PLAYER_WIDTH / 2, 0, PLAYER_HEIGHT / 2])

head_2d  = project_to_screen(world_to_local(pos_head - pos_obs, yaw, pitch), fov, W, H)
feet_2d  = project_to_screen(world_to_local(pos_feet - pos_obs, yaw, pitch), fov, W, H)
left_2d  = project_to_screen(world_to_local(left_pos - pos_obs, yaw, pitch), fov, W, H)
right_2d = project_to_screen(world_to_local(right_pos - pos_obs, yaw, pitch), fov, W, H)

print(head_2d)
print(feet_2d)
print(left_2d)
print(right_2d)

if head_2d is not None and feet_2d is not None:
    if left_2d is not None and right_2d is not None:
        x_min = int(min(left_2d[0], right_2d[0]))
        x_max = int(max(left_2d[0], right_2d[0]))
    else:
        # fallback ширина
        width_px = 40
        x_center = int(head_2d[0])
        x_min, x_max = x_center - width_px // 2, x_center + width_px // 2

    y_min = int(head_2d[1])
    y_max = int(feet_2d[1])

    # Ограничим bbox в пределах экрана
    x_min = max(0, min(x_min, W - 1))
    x_max = max(0, min(x_max, W - 1))
    y_min = max(0, min(y_min, H - 1))
    y_max = max(0, min(y_max, H - 1))

    # Отрисовка
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.circle(frame, tuple(np.int32(head_2d)), 4, (255, 0, 0), -1)
    cv2.circle(frame, tuple(np.int32(feet_2d)), 4, (0, 255, 0), -1)

    cv2.imwrite("bbox_projection_cv2.png", frame)
    print("Сохранено: bbox_projection_cv2.png")
else:
    print("Игрок вне поля зрения — bbox не строится.")
