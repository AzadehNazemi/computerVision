import cv2,os,sys
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
  
        fn = os.path.join(root, filename)
      
        img=cv2.imread(fn)
      
        (B, G, R) = cv2.split(img)
        blue=img
        blue[R==255]=0
        blue[B==255]=255
        blue[G==255]=0
        cv2.imwrite("blue/"+filename, 255-blue )
        
        red=img
        red[R==255]=255
        red[B==255]=0
        red[G==255]=0
        cv2.imwrite("red/"+filename, 255-red )
    
        green=img
        green[R==255]=0
        green[B==255]=0
        green[G==255]=255
        cv2.imwrite("green/"+filename, 255-green )
        
        yellow=img
        yellow[R==255]=0
        yellow[B==255]=255
        yellow[G==255]=255
        cv2.imwrite("green/"+filename, 255-yellow)
