import numpy as np
import sys,cv2
im =cv2.imread(sys.argv[1])
gray=cv2.imread(sys.argv[1],0)
tmp = gray > 0
coords = np.argwhere(tmp)
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1
print(x0, x1, y0, y1)
cropped = im[x0:x1, y0:y1]
cv2.imwrite("trimmed.jpg", cropped)
