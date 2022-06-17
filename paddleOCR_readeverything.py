 '''
pip3 install  paddleocr paddlepaddle paddle common data prox tight 
'''
import sys  
ocr = PaddleOCR(use_angle_cls=True)
img_path=sys.argv[1] 
result = ocr.ocr(img_path)
print (result[0][-1][0])
