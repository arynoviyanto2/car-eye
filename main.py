import cv2.cv2 as cv2
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

backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=True)
# backSub = cv2.createBackgroundSubtractorKNN()

kernel = np.ones((2, 2), np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    # plt.imshow(frame)
    # plt.show()

    fgMask = backSub.apply(frame)

    # _, fgMask = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
    #
    # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel, iterations=3)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_ERODE, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    y1 = 300
    y2 = 486
    x1 = 192
    x2 = 784
    minArea = 5000

    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        xc = x + w / 2
        yc = y + h / 2
        if (w * h > minArea) and (xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2):
            print(w * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

    cv2.imshow('FG Mask', fgMask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(10)

    if key == 27:  # Escape
        break

    if key == 32:  # Space bar
        cv2.waitKey(-1)  # wait until any key is pressed

cap.release()
cv2.destroyAllWindows()
