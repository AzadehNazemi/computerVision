from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

import imutils
import cv2,sys

class ColorLabeler:
	def __init__(self):
		colors = OrderedDict({
			"red": (255, 0, 0),
			"green": (0, 255, 0),
			"blue": (0, 0, 255)})

		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []

		for (i, (name, rgb)) in enumerate(colors.items()):
			self.lab[i] = rgb
			self.colorNames.append(name)

		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]

		minDist = (np.inf, None)

		for (i, row) in enumerate(self.lab):
			d = dist.euclidean(row[0], mean)

			if d < minDist[0]:
				minDist = (d, i)

		return self.colorNames[minDist[1]]
        



image=cv2.imread(sys.argv[1])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cl = ColorLabeler()

for c in cnts:
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)

	color = cl.label(lab, c)

	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	text = "{}".format(color)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, text, (cX, cY),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	cv2.imshow("Image", image)
	cv2.waitKey(0)
