
import cv2,os,sys,imutils
from imutils.paths import list_images
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import entropy
from math import log, e
from skimage import feature   
from imutils.paths import list_images
def solidity(img):
    gray=img
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6)) 
    graydilate = cv2.erode(gray, element) 

    ret,thresh = cv2.threshold(graydilate,127,255,cv2.THRESH_BINARY_INV)  
    imgbnbin = thresh


    contours, hierarchy = cv2.findContours(imgbnbin, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))


    Areacontours = list()
    calcarea = 0.0
    unicocnt = 0.0
    for i in range (0, len(contours)):
         area = cv2.contourArea(contours[i])
         if (calcarea<area):
            calcarea = area
            unicocnt = contours[i]

    area = cv2.contourArea(unicocnt)

    hull = cv2.convexHull(unicocnt) 
    hull_area = cv2.contourArea(hull)

    solidity = float(area)/hull_area
    return (solidity)
    
def vol(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurvol=cv2.Laplacian(gray, cv2.CV_64F).var()
    return blurvol
import skimage
def entropy (img):  
    entropy = skimage.measure.shannon_entropy(img)
    return (entropy)
for  imagePath in list_images(sys.argv[1]):
    filename=imagePath.split("/")[-1]
    image = (cv2.imread(imagePath))
    blurr=vol(image)
    image = cv2.cvtColor(    image, cv2.COLOR_BGR2GRAY)
    gcm= greycomatrix(image, [1], [0], 256, symmetric=False, normed=True) 
            
    correlation=greycoprops(gcm, 'correlation').flatten()
    homogeneity=greycoprops(gcm, 'homogeneity').flatten()
    contrast= greycoprops(gcm, 'contrast').flatten()
    energy= greycoprops(gcm, 'energy').flatten()
    entrop=entropy(image)  
    solid=solidity(image)
    print(contrast,energy,correlation,homogeneity,blurr, entrop,solid) 

    feature=(contrast,energy,correlation,homogeneity,entrop)
