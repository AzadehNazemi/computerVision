import sys
import os
import cv2
import numpy as np
import imutils

def templateMATCHING(gray, template):
    mask = np.zeros(gray.shape, dtype=np.uint8)

    (tH, tW) = template.shape[:2]
    h, w = gray.shape[:2]
    found = (0, 0)
    if gray.shape[0] < tH or gray.shape[1] < tW:
        return (w, h, 0, 0, tH, tW)

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    if found == (0, 0) or maxVal > found[0]:
        found = (maxVal, maxLoc)

    (_, maxLoc) = found
    if maxLoc != 0:
        startX, startY = (int(maxLoc[0]), int(maxLoc[1]))
        endX, endY = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
        
        return (endX, endY, startX, startY, tH, tW)
    else:

        return (w, h, 0, 0, tH, tW)


pathi = sys.argv[1]
for root, dirs, files in os.walk(pathi):
    for filename in files:

        fn = os.path.join(root, filename)
        template = cv2.imread(sys.argv[2],0)
        image = cv2.imread(fn)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (endX, endY, startX, startY, th,tw) = templateMATCHING(gray, template)
        print (endX,endY)

