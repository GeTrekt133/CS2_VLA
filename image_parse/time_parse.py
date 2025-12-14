import cv2
import time


def crop(path, cords):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[cords[1]:cords[3], cords[0]:cords[2]]
    return img

def match_digit(crop, templates):
    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    best_score = -1
    best_digit = None
    
    for digit, templ in templates.items():
        res = cv2.matchTemplate(img_bin, templ, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit, best_score

if __name__ == "__main__":
    cords = (305, 5, 335, 20)
    templates = {str(i): cv2.imread(f"templates/{i}_time.jpg", 0) for i in range(10)}
    templates["bomb1"] = cv2.imread(f"templates/bomb1.jpg", 0)
    templates["bomb2"] = cv2.imread(f"templates/bomb2.jpg", 0)
    templates["bomb3"] = cv2.imread(f"templates/bomb3.jpg", 0)
    img = crop(f"D:\\FramesDataset\\1-03f67162-abf3-437e-b575-86538acdb399-1-1\\tick_62392.jpg", cords)
    # dig1, score1 = match_digit(img[:, 4:11], templates)
    # dig2, score2 = match_digit(img[:, 13:20], templates)
    # dig3, score3 = match_digit(img[:, 19:26], templates)
    # _, digit1 = cv2.threshold(img[:, 4:11], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, digit2 = cv2.threshold(img[:, 13:20], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, digit3 = cv2.threshold(img[:, 19:26], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite("tick_62392_1.jpg", digit1)
    # cv2.imwrite("tick_62392_2.jpg", digit2)
    # cv2.imwrite("tick_62392_3.jpg", digit3)
    bomb_planted = False
    for i in range(57240, 64604, 4):
        t = time.time()
        img = crop(f"D:\\FramesDataset\\1-03f67162-abf3-437e-b575-86538acdb399-1-1\\tick_{i}.jpg", cords)
        dig1, score1 = match_digit(img[:, 4:11], templates)
        dig2, score2 = match_digit(img[:, 13:20], templates)
        dig3, score3 = match_digit(img[:, 19:26], templates)
        ts = f"{dig1}:{dig2}{dig3}"
        if any(['bomb' in x for x in [dig1, dig2, dig3]]):
            if bomb_planted == False:
                bomb_planted = True
                bomb_time = 40
        if bomb_planted:
            bomb_time -= 1/16
            ts = f"bomb - {round(bomb_time)}"
        print(f"Time: {ts}; Scores: [{score1}, {score2}, {score3}]; Time - {time.time() - t}")
