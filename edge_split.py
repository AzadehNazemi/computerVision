
import cv2,sys,os
import numpy as np            
            
def edged(img):
    blue, green, red = cv2.split(img)
    blue_edges = cv2.Canny(blue, 100, 250)
    green_edges = cv2.Canny(green, 100, 250)
    red_edges = cv2.Canny(red, 100, 250)
   
    edges = blue_edges | green_edges | red_edges
    return(edges)
                  
pathi=  sys.argv[1]   		
 
for root, dirs, files in os.walk(pathi):
    for filename in files:      
            image=cv2.imread(os.path.join(root, filename))    
            edge=edged(image)
            gray=cv2.imread(os.path.join(root, filename),0)    
            edges = cv2.Canny(image, 100, 250)
            cv2.imwrite(filename,np.hstack([gray,edges,edge]))
           
