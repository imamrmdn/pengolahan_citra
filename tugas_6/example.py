import cv2
import numpy as np
import matplotlib.pyplot as pyp

img = cv2.imread('img/angka.jpg', 0)

result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         cv2.THRESH_BINARY, 11, 2)

pyp.title('Adaptive gaussian Thresholding')
pyp.imshow(result, 'gray')


pyp.show()

cv2.waitKey(0)

cv2.destroyAllWindows()