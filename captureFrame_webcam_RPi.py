
from imutils.video import VideoStream
import datetime
import sys
import imutils
import time
import cv2

'''
RPi

camera=1

vs = VideoStream(useCamera=camera > 0).start()
'''
'''
webcam
'''
vs=cv2.VideoCapture(0)
time.sleep(2.0)

while True:

    status = vs.read()[0]
    frame = vs.read()[1]
    frame = imutils.resize(frame, width=400)

    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
