import numpy as np
import sys
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
def detect_blur_fft(image, size=60, thresh=10):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fftShift))
    plt.imsave("Magnitude.jpg",magnitude)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return (mean, mean <= thresh)


orig = cv2.imread(sys.argv[1])
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

(mean, blurry) = detect_blur_fft(gray, size=60,	thresh=20)

image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	color, 2)
print("[INFO] {}".format(text))

cv2.imshow("Output", image)
cv2.waitKey(0)
for radius in range(1, 30, 2):
    image = gray.copy()

    if radius > 0:
        image = cv2.GaussianBlur(image, (radius, radius), 0)

        (mean, blurry) = detect_blur_fft(image, size=60,
            thresh=20)

        image = np.dstack([image] * 3)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
        text = text.format(mean)
        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2)
        print("[INFO] Kernel: {}, Result: {}".format(radius, text))

    cv2.imshow("Test Image", image)
    cv2.waitKey(0)

