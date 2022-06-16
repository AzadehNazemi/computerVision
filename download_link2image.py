
import numpy as np
import urllib3 as urllib
import urllib.request
import cv2,sys

url=sys.argv[1]
print ("downloading %s" % (url))
resp = urllib.request.Request(url)
with urllib.request.urlopen(resp) as response:

    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)   

    cv2.imshow("Image", image)
    cv2.waitKey(0)
