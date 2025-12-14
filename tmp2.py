# import cv2


# img = cv2.imread(r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1\tick_4868.jpg")
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# left, top, right, bottom = (10, 25, 140, 170)
# radar = img[top:bottom, left:right, :]
# radar = cv2.resize(radar, (224, 224))
# cv2.imwrite("radar_crop.jpg", radar)

n = 0
m = 0
c = 20
i = 1
if  n + m <= 0:
    i = 0
while (n + m) // 20 != 0:
    i += 1
    n -= 20
print(i)