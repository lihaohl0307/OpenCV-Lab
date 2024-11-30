import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image as grey scale
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# apply simple thresholding
_, thresh_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.imshow(thresh_simple, cmap='gray')
plt.title('Simple Thresholding')
plt.show()

# Apply adaptive mean thresholding
thresh_adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C
, cv2.THRESH_BINARY, 11, 2)

# Apply adaptive Gaussian thresholding
thresh_adaptive_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C
,  cv2.THRESH_BINARY, 11, 2)

# Show both results
plt.subplot(1, 2, 1), plt.imshow(thresh_adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')

plt.subplot(1, 2, 2), plt.imshow(thresh_adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')

plt.show()

# Apply Otsu’s binarization
_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the result
plt.imshow(thresh_otsu, cmap='gray')
plt.title('Otsu’s Binarization')
plt.show()


