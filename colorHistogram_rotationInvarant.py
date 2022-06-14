 
from matplotlib import pyplot as plt
import numpy as np
import cv2,sys
import numpy as np
def histogram(image):
    hists = []

    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        (h, w) = image.shape[:2]
        norm =np.zeros((w,h))
        hist = cv2.normalize(hist,  norm, 0, 255, cv2.NORM_MINMAX)
        hists.append(hist)

    return hists

def rotate(image, angle):
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	affine = cv2.warpAffine(image, M, (w, h))

	return affine

def resize(image, scale):
	dim = tuple(np.int32((image.shape[0] * scale, image.shape[1] * scale)))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	return resized

image = cv2.imread(sys.argv[1])



for (i, (angle, scale)) in enumerate(((0, 1.0), (90, 0.5), (180, 0.25))):
    affine = resize(rotate(image, angle), scale)
    hists = histogram(affine)

    title = "Angle: %d, Scale: %.2f" % (angle, scale)
    cv2.imshow(title, affine)
    cv2.waitKey(0)
    for (hist, color) in zip(hists, ("b", "g", "r")):
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
        plt.ylim([0, 1])
        
        plt.savefig("hist"+str(i)+color+".jpg")
        plt.clf()
