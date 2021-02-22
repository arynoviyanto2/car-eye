import cv2
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread(os.path.join(working_frame_dir, frame_filename[200]))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, gray_img = cv2.threshold(img,225,255,cv2.THRESH_BINARY)

edges = cv2.Canny(gray_img,250,251)

plt.imshow(edges)
plt.show()

# element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# edges = cv2.dilate(edges, element)

# plt.imshow(edges)
# plt.show()

lines = cv2.HoughLines(edges, 1, np.pi / 90, 60, None, 0, 0)
    
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #if (i in [0, 2, 3, 5]):
        cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
        print(str(i) + ': ' + str(rho) + ', ' + str(theta))
        cv2.putText(img, str(i), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()