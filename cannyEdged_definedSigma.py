import os,cv2,sys
import numpy as np
def auto_canny(image):
    sigma = 0.33
    v = np.median(image)
    l   ower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged
	
image=cv2.imread  (sys.argv[1])	
cv2.imshow('edged.jpg',auto_canny(image))
cv2.waitKey(0)  
