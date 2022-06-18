from imutils import paths
import numpy as np
import sys
import imutils
import cv2


class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images
        ha,wa=imageA.shape[:2]
        imageB=cv2.resize(imageB,(wa,ha))
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None

        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        vis, xs, xe = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                       status)

        return (result, status,vis, xs, xe)

        
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        kps = np.float32([kp.pt for kp in kps])

        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))

        vis = imageA[:, np.min(ptA[0]): np.max(ptB[0])]

        print(vis.shape[:2])

        ww, hh = vis.shape[:2]
        final_answer = (2*wB)-hh
        print("final answer", (2*wB)-hh)
        return vis, np.min(ptA[0]),  final_answer
imagePaths = sorted(list(paths.list_images(sys.argv[1])))
images = []
st=Stitcher()
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)
print("[INFO] stitching images...")
if st.stitch(images)!=None:
    stitched,status,_,_,_=st.stitch(images)

    if status is not None:
        cv2.imwrite(sys.argv[2], stitched)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
    else:
        print("[INFO] image stitching failed ")
else:
    print("[INFO] image stitching failed")
