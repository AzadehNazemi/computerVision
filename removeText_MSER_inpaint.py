

import numpy as np
import imutils
import cv2
import os
import sys
from skimage import io
mser = cv2.MSER_create()
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:

        fn = os.path.join(root, filename)
        img = cv2.imread(fn)
        ho, wo = img.shape[:2]

        img1 = cv2.imread(fn, 1)
        img0 = cv2.imread(fn, 0)
        origin = img1
        ho, wo = img1.shape[:2]
        img1 = cv2.resize(img1, (int(wo/1), int(ho/1)))
        img0 = cv2.resize(img0, (int(wo/1), int(ho/1)))
        gray = img0

        ho, wo = img1.shape[:2]
        img_copy = img1.copy()

        regions, bboxes = mser.detectRegions(gray)

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        cv2.polylines(img_copy, hulls, 1, (0, 0, 0))

        mask = np.zeros((img1.shape[0], img1.shape[1], 1), dtype=np.uint8)

        for contour in hulls:

            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -5)
        out1 = filename.replace(".jpg", "_text.jpg")
        only = cv2.bitwise_and(img0, img0, mask=mask)
        out2 = filename.replace(".jpg", "_noText.jpg")
        out3 = filename.replace(".jpg", "_filled.jpg")
        cv2.imwrite(out2, mask)

        radius = 0.1
        flags = cv2.INPAINT_TELEA
        threshMap = cv2.threshold(
            only, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        mask = threshMap
        
        text = cv2.bitwise_and(img1, img1, mask=mask)
        cv2.imwrite(out1, text)
        mask = cv2.imread(out2)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ho, wo = origin.shape[:2]

        gray10 = cv2.resize(gray, (wo, ho))

        output = cv2.inpaint(origin, gray10, radius, flags=flags)
        cv2.imwrite(out3, output)
        
