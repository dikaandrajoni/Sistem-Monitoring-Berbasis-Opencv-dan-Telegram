import time
import cv2
import numpy as np
import glob
import os
import time
import pygame

cam = cv2.VideoCapture(0)
cam.set(3, 640) #Set video width
cam.set(4, 480) #Set video height

kernal = np.ones((5,5),np.uint8)
firstFrame = None

while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1) #flip Vertical
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if firstFrame is None:
        firstFrame = gray
        continue

    diffImage = cv2.absdiff(firstFrame, gray)
    blurImage = cv2.GaussianBlur(diffImage, (21,21),0)
    _, thresholdImage = cv2.threshold(blurImage, 20, 255, cv2.THRESH_BINARY)
    dilateImage = cv2.dilate(thresholdImage, kernal, iterations=5)

    contours, _ = cv2.findContours(dilateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 1000:
            cv2.rectangle(img, (x,y),(x+w,y+h),(255,255,0),1)


    cv2.imshow("Monitoring", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

    

    


    
