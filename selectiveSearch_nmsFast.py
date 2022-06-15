import numpy as np
import sys
import imutils
import cv2

def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick].astype("int")
image = cv2.imread(sys.argv[1]) 
image = imutils.resize(image, width=600)
ho, wo = image.shape[:2]

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()

rects = ss.process()

output = image.copy()
orig=image.copy () 
for i in range(0, len(rects), 100):

    boundingBoxes=[]
    for (x, y, w, h) in rects[i:i + 100]:
           startX=x
           startY=y
           endX=x+w
           endY=y+h
           cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
           boundingbox=(startX,startY,endX,endY)
           boundingBoxes.append(boundingbox)
    boundingBoxes=np.array(boundingBoxes)
    pick = non_max_suppression_fast(boundingBoxes, 0.3)
    

    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)

       

cv2.imwrite("after_nms.png",output)
cv2.imwrite("before_nms.png",orig)

