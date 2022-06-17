
import imutils
import cv2,sys

class RGBHistogram:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])

		if imutils.is_cv2():
			hist = cv2.normalize(hist)

		else:
			hist = cv2.normalize(hist,hist)

		return hist.flatten()
image=cv2.imread(sys.argv[1])
ch=RGBHistogram([8,8,8])
print(ch.describe(image))
