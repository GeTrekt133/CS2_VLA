import cv2


img = cv2.imread("C:\\Users\\misas\\CS2_NN\\dataset\\frames\\round\\dust2\\img30940.jpg")
time_crop = img[5:24, 220:260]
digit1 = cv2.resize(time_crop[:, 2:13], (28, 28))
digit2 = cv2.resize(time_crop[:, 17:28], (28, 28))
digit3 = cv2.resize(time_crop[:, 27:-2], (28, 28))
cv2.imwrite('round_time.jpg', time_crop)
cv2.imwrite('digit1.jpg', digit1)
cv2.imwrite('digit2.jpg', digit2)
cv2.imwrite('digit3.jpg', digit3)