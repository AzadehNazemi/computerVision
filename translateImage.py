import numpy as np
import sys
import imutils
import cv2

image = cv2.imread(sys.argv[1])
'''
shift the image 50 pixels to the right 
shift the image 100 pixels to the down
'''
M = np.float32([[1, 0, 50], [0, 1, 100]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imwrite("ShiftedDownRight.jpg", shifted)

'''
shift the image 50 pixels to the left 
shift the image 100 pixels to the up
'''
M = np.float32([[1, 0, -50], [0, 1, -100]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imwrite("ShiftedUpLeft.jpg", shifted)

'''
shift the image 50 pixels to the right 
shift the image 100 pixels to the down
'''

shifted = imutils.translate(image, 50, 100)
cv2.imwrite("ShiftedRightDown.jpg", shifted))
