import numpy as np
import sys
import imutils
import cv2


image = cv2.imread(sys.argv[1]) 
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) =saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

cv2.imwrite("SpectralResidual.jpg", saliencyMap)


saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imwrite("FineGrained.jpg", saliencyMap)
cv2.imwrite("Thresh.jpg", threshMap)
'''
max_detections=10
model="ObjNessB2W8HSV.idx.yml/ObjNessB2W8HSV.idx.yml"
saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath(model)
    (success, saliencyMap) = saliency.computeSa li   ency(image)
numDetections = saliencyMap.shape[0]
for i in range(0, min(numDetections, max_detections)):
	(startX, startY, endX, endY) = saliencyMap[i].flatten()
	
	output = image.copy()
	color = np.random.randint(0, 255, size=(3,))
	color = [int(c) for c in color]
	cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
	cv2.imwrite("Image.jpg", output)
	
from imutils.video import VideoStream
import imutils
import cv2
saliency = None
vs = VideoStream(src=0).start()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	if saliency is None:
		saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
		saliency.setImagesize(frame.shape[1], frame.shape[0])
		saliency.init()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(success, saliencyMap) = saliency.computeSaliency(gray)
	saliencyMap = (saliencyMap * 255).astype("uint8")
	cv2.imwrite("Frame.jpg", frame)
	cv2.imwrite("Map.jpg", saliencyMap)
'''
