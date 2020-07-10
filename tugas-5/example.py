import cv2
import numpy as np

img = cv2.imread('./images/ballnew.jpg')

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([94, 80, 2])
upper_blue = np.array([126, 255, 255])

mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('img', img)
cv2.imshow('mask_image', mask)
cv2.imshow('final_result', result)
cv2.waitKey(0)

cv2.destroyAllWindows()
