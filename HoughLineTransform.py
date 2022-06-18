

import numpy as np
import cv2
import os
import sys


img= cv2.imread(sys.argv[1])
h, w = img.shape[:2]
edges = cv2.Canny(img, 150, 150, apertureSize=3)
minLineLength = int(w/20)
lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                        lines=np.array([]), minLineLength=minLineLength, maxLineGap=int(h/5))
a, b, c = lines.shape
paralell = []
gray=img
for i in range(1, a):
    if lines[i][0][1] == lines[i][0][3]:
      paralell.append(lines[i][0][1])
paralell = sorted(paralell)
for ii in range(len(paralell)):

    cv2.line(img,  (0, paralell[ii]), (w, paralell[ii]),
             (0, 255, 0), 1, cv2. cv2.LINE_AA)
    print(paralell[ii])
    cv2.line(img, (0, paralell[ii]), (w, paralell[ii]),
             (255, 255, 255), 1, cv2. cv2.LINE_AA)

cv2.imwrite('hough.png', img)

