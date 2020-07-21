##Image Segmentation Using Background Subtraction
import cv2
import numpy as npy
import matplotlib.pyplot as pyp

cap = cv2.VideoCapture('assets/walking.mp4')
kernel_dil = npy.ones((20,20), npy.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
##background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, video = cap.read()

    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    if ret == True:
        fgmask = fgbg.apply(video)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        delasi = cv2.dilate(fgmask, kernel_dil, iterations=2)
        contur, _ = cv2.findContours(delasi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # drawbox object
        for obj in contur:
            x, y, w, h = cv2.boundingRect(obj)
            if cv2.contourArea(obj) < 4355:
                continue
            cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(video, "Status : {}".format('Walk and Move'),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # add description height and width object
            cv2.putText(video, "w={},h={}".format(w,h), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

        cv2.imshow('feed', video)
        cv2.imshow('fgmask', fgmask)
    
    if cv2.waitKey(30) == 27:
        break

pyp.title('Object detection')
cap.release()
cv2.destroyAllWindows()