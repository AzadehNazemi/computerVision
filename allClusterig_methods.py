import matplotlib.pyplot as plt
import scipy
from skimage.metrics import structural_similarity as ssim
from skimage import feature
import cv2
import os
import sys
import imutils
import numpy as np
from imutils import paths
from PIL import Image
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


def c10_GAUSSIAN(X):

    model = GaussianMixture(n_components=2)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c9_SPECTRAL(X):

    model = SpectralClustering(n_clusters=2)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c8_OPTICS(X):
    model = OPTICS(eps=0.8, min_samples=2)
    yhat = model.fit_predict(X)

    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c7_MEANSHIFT(X):
    model = MeanShift()
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c6_MININBATCH(X):
    model = MiniBatchKMeans(n_clusters=2)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c5_KMEANS(X):
    model = KMeans(n_clusters=2)
    model.fit(X)

    yhat = model.predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c4_BRICH(X):
    model = Birch(threshold=0.01, n_clusters=2)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c3_AGGLOMERE(X):

    model = AgglomerativeClustering(n_clusters=2)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show


def c2_AFFINITY(X):
    model = AffinityPropagation(damping=0.9)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()


def c1_DBSCAN(X):

    model = DBSCAN(eps=0.30, min_samples=2)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])

    plt.show()


def GCLM(imageB):
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    gcm = greycomatrix(grayB, [1], [0], 256, symmetric=False, normed=True)
    corrolation = np.float(greycoprops(gcm, 'correlation').flatten())*1000
    homogeneity = np.float(greycoprops(gcm, 'homogeneity').flatten())*10000
    contrast = np.float(greycoprops(gcm, 'contrast').flatten())*1
    energy = np.float(greycoprops(gcm, 'energy').flatten())*10000
    data = [contrast, energy, homogeneity, corrolation]

    return data


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


class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):

        hist = cv2.calcHist([image], [0, 1, 2],
                            None, self.bins, [0, 256, 0, 256, 0, 256])



        return hist.flatten()


def histogram(image):
    hists = []

    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        hists.append(hist)

    hists = np.array(hists)

    hists = hists.flatten()
    return np.array(hists)


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    return d


def HOG(imageB):
    imageB = cv2.resize(imageB, (200, 200)).astype("float32")
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (HB, hogImage) = feature.hog(grayB, orientations=9, pixels_per_cell=(10, 10),
                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    return(HB)


def LBP(imageB):
    imageB = cv2.resize(imageB, (200, 200)).astype("float32")
    desc = LocalBinaryPatterns(24, 8)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    histB = desc.describe(grayB)
    return (histB)


def allCLUSTER(X):
    c2_AFFINITY(X)
    c3_AGGLOMERE(X)
    c4_BRICH(X)
    c5_KMEANS(X)
    ''' 
    5
    '''
    c6_MININBATCH(X)
    c7_MEANSHIFT(X)
    c8_OPTICS(X)
    c9_SPECTRAL(X)
    c10_GAUSSIAN(X)


def clustering(X, Xrelevent):
    model = KMeans(n_clusters=2)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0

    for cluster in clusters:
        row_ix = where(yhat == cluster)

        for ii in row_ix[0]:
            img = cv2.imread(Xrelevent[ii][1])

            print(Xrelevent[ii], cluster)
            cv2.imwrite(str(cluster)+"\\"+Xrelevent[ii][0], img)
            if int(cluster) == 0:
                c0 = c0+1
            if int(cluster) == 1:
                c1 = c1+1
            if int(cluster) == 2:
                c2 = c2+1
            if int(cluster) == 3:
                c3 = c3+1
            if int(cluster) == 4:
                c4 = c4+1
            if int(cluster) == 5:
                c5 = c5+1


    x = np.array(["cluster0", "cluster1", "cluster2",
                  "cluster3", "cluster4", "cluster5"])
    y = np.array([c0, c1, c2, c3, c4, c5])
    plt.bar(x, y, width=0.5, color=['red', 'pink',
                                    'blue', 'green', 'cyan', 'olive'])
    plt.xlabel("Texture labels")
    plt.ylabel('Importance')
    plt.savefig("clustering_bar_chart.png")
    plt.show()


listed = list(paths.list_images(sys.argv[1]))
imagePaths = sorted(listed, key=lambda e: e)

XX = []
YY = []
ZZ = []
GG = []
GGrelevent = []
for imagePath in imagePaths:
    filenameB = imagePath.split(os.path.sep)[-1]
    nameimageB = imagePath
    imageB = cv2.imread(nameimageB)
    GM = GCLM(imageB)
    HB = HOG(imageB)
    LM = LBP(imageB)
    properties = np.hstack([GM, HB, LM])
    print(filenameB)
    GGrelevent.append([filenameB, imagePath])
    GG.append(properties)

GG = np.array(GG)
GGrelevent = np.array(GGrelevent)
allCLUSTER(GG)
clustering(GG, GGrelevent)

