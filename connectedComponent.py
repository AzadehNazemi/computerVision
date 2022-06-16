import numpy as np
import cv2
import sys
image = cv2.imread(sys.argv[1])
connectivity =4
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

output = cv2.connectedComponentsWithStats(
	thresh, connectivity, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
output = image.copy()

componentMask=np.zeros(image.shape[:2], dtype="uint8")

for i in range(1, numLabels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    cv2.circle(componentMask, (int(cX), int(cY)), 4,255, -1)
cv2.imshow("ConnectedComponentMask", componentMask)
cv2.waitKey(0)
cv2.imshow("Output", output)
cv2.waitKey(0)
