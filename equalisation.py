import sys
import cv2

image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
cv2.imshow("HistogramEqualisation", equalised)
cv2.waitKey(0)
