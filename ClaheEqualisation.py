import sys
import cv2

image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tile=8
clip=2.0
clahe = cv2.createCLAHE(clipLimit=clip,tileGridSize=(tile,tile))
equalised = clahe.apply(gray)
cv2.imshow("CLAHE", equalised)
cv2.waitKey(0)
