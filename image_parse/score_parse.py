import cv2
import numpy as np


def crop(path, cords):
    img = cv2.imread(path)
    # img = replace_colors(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[cords[1]:cords[3], cords[0]:cords[2]]
    return img

def replace_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([180, 180, 180])
    upper_blue = np.array([230, 230, 230])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_orange = np.array([180, 180, 150])
    upper_orange = np.array([240, 240, 190])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.bitwise_or(mask_blue, mask_orange)
    img[mask > 0] = [255, 255, 255]

    return img

def match_digit(img, templates):
    score_ct = img[2:13, :11]
    score_t = img[2:13, 18:29]
    _, score_ct = cv2.threshold(score_ct, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, score_t = cv2.threshold(score_t, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # return score_ct, score_t
    scores = []
    for score in [score_ct, score_t]:
        best_score = 0
        for digit, templ in templates.items():
            res = cv2.matchTemplate(score, templ, cv2.TM_CCOEFF_NORMED)
            res_score = res.max()
            if res_score > best_score:
                best_score = res_score
                best_digit = digit
        scores.append([best_digit, best_score])
    return scores


if __name__ == "__main__":
    cords = (305, 25, 335, 40)
    img = crop(r"D:\FramesDataset\1-063bac12-9823-4269-8acc-205642238a8f-1-1\tick_100000.jpg", cords)
    img2 = img[2:13, :11]
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite("score.jpg", img2)
    templates = {str(i): cv2.imread(f"templates/score_{i}.jpg", 0) for i in range(19)}
    scores = match_digit(img, templates)
    print(scores)
    # cv2.imwrite("score_ct.jpg", score_ct)
    # cv2.imwrite("score_t.jpg", score_t)