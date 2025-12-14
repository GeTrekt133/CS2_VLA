import cv2
import os


def crop(path, cords):
    img = cv2.resize(cv2.imread(path), (640, 640))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[cords[1]:cords[3], cords[0]:cords[2]]
    return img

if __name__ == "__main__":
    cords = (235, 115, 405, 145)
    # img = cv2.resize(cv2.imread(r"C:\Users\misas\CS2_NN\frames\match1\frame_0000081.jpg"), (640, 640))
    tampletes = [(cv2.imread(os.path.join("templates", t)), t) for t in os.listdir("templates") if t.startswith("round_end")]
    root = r"C:\Users\misas\CS2_NN\frames\match1"
    for img_name in os.listdir(root):
        img = crop(os.path.join(root, img_name), cords)
        scores = []
        for tpl in tampletes:
            res = cv2.matchTemplate(img, tpl[0], cv2.TM_CCOEFF_NORMED)
            scores.append(res.max())
        # find index of the best matching template and its score
        if len(scores) == 0:
            continue
        max_score = max(scores)
        max_index = scores.index(max_score)
        if max_score > 0.8:
            tpl_name = tampletes[max_index][1]
            print(f"Found round end in {img_name} with score {max_score}, template {tpl_name}")
    # cv2.imwrite("round_end.jpg", img)