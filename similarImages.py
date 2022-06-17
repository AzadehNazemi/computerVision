


import numpy as np
import cv2 as cv
import cv2
import math
import os
import sys

MIN_MATCH_COUNT = 4

referenceMatrix = np.array([
    [1, 0, 0],
    [0, 1, 0], 
    [0, 0, 1],
])
img_compare="compare_image.jpg"
def compare_image(img1src, img2src):
    img1 = cv2.imread(img1src,0)
    
    if img1.shape[1] < img1.shape[0]:
        img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        

    img2 = cv2.imread(img2src,0)

    
    similarity = 10000

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        similarity = abs(np.sum(M - referenceMatrix) / M.size)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        
        print( "Images are similar, enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
 
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        
        matchesMask=None
    draw_params = dict(matchColor = (0,255,0), 
                       singlePointColor = None,
                       matchesMask = matchesMask, 
                       flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imwrite(img_compare, img3)
    
    return similarity

compare_image(sys.argv[1],sys.argv[2])
    




