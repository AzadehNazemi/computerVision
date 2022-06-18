from skimage.metrics import structural_similarity as ssim
   
import cv2,os,sys,imutils,pickle
import numpy as np          
from PIL import Image
import imagehash

cutoff =5
    
def histogram(image):       
    hists = []

    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist,hist)
        hists.append(hist)

    return hists    

def chi2_distance( histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		return d

nameimageA=sys.argv[1]
nameimageB=sys.argv[2]

filenameA=nameimageA.split(os.path.sep)[-1]
filenameB=nameimageB.split(os.path.sep)[-1]
imageA=cv2.imread(nameimageA)
imageB=cv2.imread(nameimageB)
H,W=imageA.shape[:2]
imageB=cv2.resize(imageB,(W,H))

hash0 = imagehash.average_hash(Image.open(nameimageA))
hash1 = imagehash.average_hash(Image.open(nameimageB)) 

D=chi2_distance(histogram(imageA),histogram(imageB))
err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
err /= float(imageA.shape[0] * imageA.shape[1])
similarity =ssim(imageA,imageB, multichannel=True)
L=filenameB+", "+filenameA+" ,"+str(err)+','+str(similarity)+','+str(D)+","+str(hash0 - hash1)
print(L)
