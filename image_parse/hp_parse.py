import cv2
import os


def crop(path, cords):
    img = cv2.resize(cv2.imread(path), (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[cords[1]:cords[3], cords[0]:cords[2]]
    return img


if __name__ == "__main__":
    cords = (168, 600, 200, 619)
    img = crop(r"C:\Users\misas\CS2_NN\frames\match2\frame_0000074.jpg", cords)
    _, digit1 = cv2.threshold(img[:, 1:10], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, digit2 = cv2.threshold(img[:, 10:19], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, digit3 = cv2.threshold(img[:, 20:28], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("hp_1.jpg", digit1)
    cv2.imwrite("hp_2.jpg", digit2)
    cv2.imwrite("hp_3.jpg", digit3)
    # cv2.imwrite("hp.jpg", img)