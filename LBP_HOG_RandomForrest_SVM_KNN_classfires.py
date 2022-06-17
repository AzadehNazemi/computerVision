from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from imutils import paths
import numpy as np
import pickle
import os
import cv2,sys





def HOG(gray):
    features = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")       

    return features

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):

        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


desc = LocalBinaryPatterns(24, 8)   
data = []
labels = []

for imagePath in paths.list_images(sys.argv[1]):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100)).astype("float32")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist=HOG(gray)
    descript = desc.describe(gray)
    label = int((imagePath.split(os.path.sep)[-1]).split("_")[0])
    
    print(label)
    data.append( np.hstack([hist, descript]))   
    labels.append(label)
model1 = RandomForestClassifier(n_estimators=100)
model2 =LinearSVC(C=1000.0, random_state=42)
model3=KNeighborsClassifier()

model1.fit(np.array(data), labels)
model2.fit(np.array(data), labels)
model3.fit(np.array(data), labels)

pickle.dump(model1, open('LBP_HOG_RandomForest.sav', 'wb'))
pickle.dump(model2, open('LBP_HOG_LinearSVC.sav', 'wb'))
pickle.dump(model3, open('LBP_HOG_KNeighbors.sav', 'wb'))
