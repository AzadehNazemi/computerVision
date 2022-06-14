
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import sys
import imutils
import cv2
imagePaths = sorted(list(list_images(sys.argv[1])))

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    text = "Low contrast: No"
    color = (0, 255, 0)

    if is_low_contrast(gray, fraction_threshold=0.35):
        text = "Low contrast: Yes"
        color = (0, 0, 255)

    else:
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0) 
