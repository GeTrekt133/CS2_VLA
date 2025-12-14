import cv2

# Загружаем картинку
img = cv2.imread("C:\\Users\\misas\\CS2_NN\\dataset\\frames\\round\\dust2\\img30940.jpg")

# Кроп счёта
score_crop = img[25:50, 216:268]

# Вырезаем скоры для Т и КТ
scoreCT = cv2.resize(score_crop[:, :20], (28, 28))
scoreT = cv2.resize(score_crop[:, 25:], (28, 28))

# Вырезаем отдельные циферки если счет меняется на двузначный
digit1_score2 = scoreT[:, :14]
digit2_score2 = scoreT[:, 14:]

cv2.imwrite('round_score.jpg', score_crop)
cv2.imwrite('scoreCT2.jpg', scoreCT)
cv2.imwrite('scoreT2.jpg', scoreT)
cv2.imwrite('scoreT2d1.jpg', digit1_score2)
cv2.imwrite('scoreT2d2.jpg', digit2_score2)
