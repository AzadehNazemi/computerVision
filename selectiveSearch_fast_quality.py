
import sys
import cv2
'''
["fast", "quality"]
'''
image = cv2.imread(sys.argv[1])

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ss.switchToSelectiveSearchQuality()
rects = ss.process()

cc=0
for i in range(0, len(rects), 100):
    output = image.copy()

    for (x, y, w, h) in rects[i:i + 100]:
        cc=(cc+100)%255
        color = [cc for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

cv2.imshow("Output", output)
cv2.waitKey(0) 
