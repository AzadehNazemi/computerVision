from imutils import paths
import numpy as np
import sys
import imutils
import cv2
imagePaths = sorted(list(paths.list_images(sys.argv[1])))
images = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
if status == 0:
	cv2.imwrite(sys.argv[2], stitched)
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
else:
	print("[INFO] image stitching failed ({})".format(status))
