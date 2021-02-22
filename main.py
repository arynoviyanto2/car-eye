import cv2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import math


cap = cv2.VideoCapture("footage/sherbrooke_video.avi")
# cap = cv2.VideoCapture("footage/highway.mp4")

if not cap.isOpened():
    print("Cannot read video")
    exit()

backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=100, detectShadows=True)
#backSub = cv2.createBackgroundSubtractorKNN()

kernel = np.ones((3,3),np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    fgMask = backSub.apply(frame)

    _, fgMask = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)

    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, hierarchy	= cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)


    cv2.imshow('FG Mask', fgMask)
    cv2.imshow("frame", frame)



    key = cv2.waitKey(10)

    if key == 27: # Escape
        break

    if key == 32: # Space bar
        cv2.waitKey(-1) #wait until any key is pressed

cap.release()
cv2.destroyAllWindows()

