import cv2.cv2 as cv2
import os
import numpy as np
import argparse

from utils import assign_cars


def analyse_footage(video_filename, roi_option, sample_background_filename=None,
                    footage_dir="footage", frame_dir="frames"):
    cap = cv2.VideoCapture(os.path.join(footage_dir, video_filename))
    # cap = cv2.VideoCapture("footage/highway.mp4")

    if sample_background_filename is not None:
        background = cv2.imread(os.path.join(frame_dir, sample_background_filename))
    else:
        background = None

    if not cap.isOpened():
        print("Cannot read video")
        exit()

    backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=True)

    kernel = np.ones((2, 2), np.uint8)
    cars = []

    # prevXc = 0
    # prevYc = 0
    tick = 0

    delay = 10

    while True:
        ret, frame = cap.read()
        tick = tick + 1

        if not ret:
            print("Can't receive frame")
            break

        if background is None:
            fg_mask = backSub.apply(frame)
            # _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        else:
            fg_mask = cv2.absdiff(background, frame)
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(fg_mask, 15, 255, cv2.THRESH_BINARY)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        y1 = roi_option['y1']
        y2 = roi_option['y2']
        x1 = roi_option['x1']
        x2 = roi_option['x2']
        minArea = roi_option['minArea']

        valid_contours = contours

        # valid_contours = []
        # for i, contour in enumerate(contours):
        #     [cx1, cy1, cw1, ch1] = cv2.boundingRect(contour)
        #     is_valid = True
        #     for j, friend in enumerate(contours):
        #         if i != j:
        #             [cx2, cy2, cw2, ch2] = cv2.boundingRect(friend)
        #             if (cx1 > cx2 and cy1 > cy2 and
        #                     (cx1 + cw1) < (cx2 + cw2) and (cy1 + ch1) < (cy2 + ch2)):
        #                 is_valid = False
        #                 break
        #
        #     if is_valid:
        #         valid_contours.append(contour)

        current_cars = []
        for contour in valid_contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            xc = int(x + w / 2)
            yc = int(y + h / 2)
            if (w * h > minArea) and (xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2):
                # deltaX = xc - prevXc + np.finfo(float).eps
                # deltaY = yc - prevYc + np.finfo(float).eps

                # gradient = 1.0 * deltaY / deltaX
                # movement = deltaX * deltaX + deltaY * deltaY
                # print("{0}, {1}: tick: {4}, grad = {2:.2f}, move = {3}".format(xc, yc, gradient, movement, tick))

                # prevYc = yc
                # prevXc = xc

                car = {
                    'id': -1,
                    'center': (xc, yc),
                    'rect': (x, y, w, h),
                    'tick': tick,
                    'active': False
                }

                current_cars.append(car)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 1)

        assign_cars(cars, current_cars)

        for car in cars:
            if car['active']:
                [x, y, w, h] = car['rect']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

                cv2.putText(frame, str(car['id']), car['center'],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                # cv2.drawContours(frame, [contour], 0, (0, 255, 0), 1)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

        cv2.putText(frame, "Number of cars: {0}".format(len(cars)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('FG Mask', fg_mask)
        cv2.imshow("frame", frame)

        key = cv2.waitKey(delay)

        if key == 27:  # Escape
            break

        if key == 32:  # Space bar
            cv2.waitKey(-1)  # wait until any key is pressed

        if key == ord('s'):
            delay = 10 * delay

        if key == ord('f'):
            delay = int(delay / 10)
            delay = 1 if delay < 1 else delay

    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('--v', required=True,
                    help="Video filename")
parser.add_argument('--v_dir', default='footage',
                    help="Footage directory")
parser.add_argument('--b', default=None,
                    help="Predefined background")
parser.add_argument('--i_dir', default='frames',
                    help="Frame directory")

# python3 main.py --v sherbrooke_video.avi --b sherbrooke_frames/00000222.jpg
if __name__ == "__main__":
    # video_filename = "sherbrooke_video.avi"
    # sample_background_filename = "sherbrooke_frames/00000222.jpg"

    args = parser.parse_args()

    roi = {
        'y1': 300,
        'y2': 486,
        'x1': 192,
        'x2': 784,
        'minArea': 6000
    }

    analyse_footage(args.v, roi, args.b, footage_dir=args.v_dir, frame_dir=args.i_dir)
