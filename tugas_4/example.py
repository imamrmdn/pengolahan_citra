import cv2

img = cv2.imread('kakek.jpg', 1)

print(img)

cv2.imshow('gambar', img)
cv2.waitKey(0)
cv2.destroyAllwindows()