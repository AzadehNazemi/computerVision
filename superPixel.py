
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import sys,cv2

image = img_as_float(io.imread(sys.argv[1]  ))

for numSegments in (100, 200, 300):
    segments = slic(image, n_segments = numSegments, sigma = 5)

    cv2.imshow("Superpixels -- %d segments" % (numSegments),mark_boundaries(image, segments))
    
    cv2.waitKey(0 )
