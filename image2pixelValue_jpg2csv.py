import cv2
import os
import sys
import imutils
import numpy as np

def append2csv(csvName,Line):
    with open(csvName,"a") as file_object:
        file_object.write(Line)

print("[INFO] loading images...")
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        fn = os.path.join(root, filename)
        img = cv2.imread(fn)
        csvName= filename.replace("jpg", "csv")
        rows, cols = img.shape[:2]
        with open(csvName,"a") as file_object:
         for i in range(rows):
            for j in range(cols):
              
                if img[i, j][0] != 0 or img[i, j][1] != 0 or img[i, j][2] != 0:              
               
                    L = str(i)+","+str(j)+"," + str(img[i, j][0])+","+str(img[i, j][1]) + ","+str(img[i, j][2])+'\n'
                    file_object.write(L)
     
