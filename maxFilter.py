
import cv2,sys,os
import numpy as np            
            
                  
pathi=  sys.argv[1]   		
 
for root, dirs, files in os.walk(pathi):
    for filename in files:      
            image=cv2.imread(os.path.join(root, filename))    
            orig=image
                
            (B, G, R) = cv2.split(image)

            M = np.maximum(np.maximum(R, G), B)
            R[R < M] = 0
            G[G < M] = 0
            B[B < M] = 0
            FILTER=cv2.merge([B, G, R])
            
            cv2.imwrite(filename,np.hstack([orig,FILTER]))
           
