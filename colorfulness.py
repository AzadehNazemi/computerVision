# derive the "colorfulness" metric from folder of images  
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys ,os
def image_colorfulness(image):
	(B, G, R) = cv2.split(image.astype("float"))

	rg = np.absolute(R - G)

	yb = np.absolute(0.5 * (R + G) - B)

	(rgMean, rgStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))

	stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

	return stdRoot + (0.3 * meanRoot)




results = []

for imagePath in paths.list_images(sys.argv[1]):
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=250)
	C = image_colorfulness(image)

	cv2.putText(image, "{:.2f}".format(C), (40, 40), 
		cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

	results.append((image, C))

print("[INFO] displaying results...")
results = sorted(results, key=lambda x: x[1], reverse=True)
mostColor = [r[0] for r in results[:25]]
leastColor = [r[0] for r in results[-25:]][::-1]

mostColorMontage = build_montages(mostColor, (128, 128), (5, 5))
leastColorMontage = build_montages(leastColor, (128, 128), (5, 5))

cv2.imshow("Most Colorful", mostColorMontage[0])
cv2.waitKey(0)

cv2.imshow("Least Colorful", leastColorMontage[0])

cv2.waitKey(0)
