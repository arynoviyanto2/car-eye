import cv2.cv2 as cv2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from utils import assign_cars

cap = cv2.VideoCapture("footage/sherbrooke_video.avi")
# cap = cv2.VideoCapture("footage/highway.mp4")

if not cap.isOpened():
    print("Cannot read video")
    exit()

backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=True)
# backSub = cv2.createBackgroundSubtractorKNN()

kernel = np.ones((2, 2), np.uint8)
cars = []

prevXc = 0
prevYc = 0
tick = 0

while True:
    ret, frame = cap.read()
    tick = tick + 1

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

    y1 = 300
    y2 = 486
    x1 = 192
    x2 = 784
    minArea = 5000

    current_cars = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        xc = int(x + w / 2)
        yc = int(y + h / 2)
        if (w * h > minArea) and (xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2):
            deltaX = xc - prevXc + np.finfo(float).eps
            deltaY = yc - prevYc + np.finfo(float).eps

            gradient = 1.0 * deltaY / deltaX

            movement = deltaX * deltaX + deltaY * deltaY
            # print("{0}, {1}: tick: {4}, grad = {2:.2f}, move = {3}".format(xc, yc, gradient, movement, tick))
            prevYc = yc
            prevXc = xc
            car = {
                'id': -1,
                'center': (xc, yc),
                'rect': (x, y, w, h),
                'tick': tick,
                'active': False
            }

            current_cars.append(car)

    assign_cars(cars, current_cars)

    for car in cars:
        if car['active']:
            [x, y, w, h] = car['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv2.putText(frame, str(car['id']), car['center'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # cv2.drawContours(frame, [contour], 0, (0, 255, 0), 1)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

    cv2.putText(frame, "Number of cars: {0}".format(len(cars)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(10)

    if key == 27:  # Escape
        break

    if key == 32:  # Space bar
        cv2.waitKey(-1)  # wait until any key is pressed

cap.release()
cv2.destroyAllWindows()
