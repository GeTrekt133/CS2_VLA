import cv2


def crop(path, cords):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[cords[1]:cords[3], cords[0]:cords[2]]
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_bin


if __name__ == "__main__":
    cords = (305, 590, 335, 630)
    # crop_img = crop(r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1\tick_5308.jpg", cords)
    img = cv2.imread("img1.jpg")
    img = cv2.resize(img[:2160, :3840], (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
    _, img_bin = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("avatar1.jpg", img_bin)