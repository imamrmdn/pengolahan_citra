# Tugas Pengolahan Citra ke 8
# Nama: Imam Ramadhan
# Nim: 1644190065
# Morphological Transformation and Image Smoothing		
import cv2
import numpy as np 
import matplotlib.pyplot as pyp

img = cv2.imread('img/Screenshot.png', 0)

kernel = np.ones((5,5), np.uint8)

#remove noise
dilation = cv2.dilate(img, kernel, iterations=3)
erotion = cv2.erode(dilation, kernel, iterations=3)

#blur image
blur = cv2.blur(erotion, (5,5), 0)

pyp.subplot(121), pyp.imshow(img, 'gray'), pyp.title('Original')
pyp.xticks([]), pyp.yticks([])
pyp.subplot(122), pyp.imshow(blur, 'gray'), pyp.title('Blurred')
pyp.xticks([]), pyp.yticks([])

pyp.show()