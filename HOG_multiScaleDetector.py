import imutils
import cv2
import sys
win_stride=(8,8)
scale=1.05
padding=(16,16)
mean_shift=-1
meanShift = True if mean_shift > 0 else False

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(sys.argv[1])
image = imutils.resize(image, width=min(400, image.shape[1]))
(rects, weights) = hog.detectMultiScale(image, winStride=win_stride,
	padding=padding, scale=scale, useMeanshiftGrouping=meanShift)

for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
import imutils
import cv2
import sys
win_stride=(8,8)
scale=1.05
padding=(16,16)
mean_shift=-1
meanShift = True if mean_shift > 0 else False

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(sys.argv[1])
image = imutils.resize(image, width=min(400, image.shape[1]))
(rects, weights) = hog.detectMultiScale(image, winStride=win_stride,
	padding=padding, scale=scale, useMeanshiftGrouping=meanShift)

for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
import imutils
import cv2
import sys
win_stride=(8,8)
scale=1.05
padding=(16,16)
mean_shift=-1
meanShift = True if mean_shift > 0 else False

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(sys.argv[1])
image = imutils.resize(image, width=min(400, image.shape[1]))
(rects, weights) = hog.detectMultiScale(image, winStride=win_stride,
	padding=padding, scale=scale, useMeanshiftGrouping=meanShift)

for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
