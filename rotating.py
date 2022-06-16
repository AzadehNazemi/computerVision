import sys
import os
import cv2
import numpy as np
import imutils
image = cv2.imread(sys.argv[1])
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated45", rotated)
cv2.waitKey(0)
            
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated-90", rotated)
cv2.waitKey(0)

M = cv2.getRotationMatrix2D((10, 10), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))


cv2.imshow("RotatedArbitraryPoint", rotated)
cv2.waitKey(0)

rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated180", rotated)
cv2.waitKey(0)

rotated = imutils.rotate_bound(image, -33)
cv2.imshow("RotatedWithoutCropping", rotated)
cv2.waitKey(0)
