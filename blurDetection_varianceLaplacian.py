import cv2,os,sys,imutils
from imutils import paths
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

threshold=100															
for imagePath in paths.list_images(sys.argv[1]):

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    

    if fm < threshold:
        print(fm,imagePath,"Blurry")
    else:
        print(fm,imagePath,"Not Blurry")
