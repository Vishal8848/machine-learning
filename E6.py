# E6 - Image Segmentation : K Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread("./apple.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 0.99)

# K = 2
retval, labels, centers = cv.kmeans(pixels, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))

plt.imshow(segmented_image)
plt.show()