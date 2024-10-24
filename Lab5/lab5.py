import cv2
import numpy as np
import matplotlib.pyplot as plt

# “0” in cv2.imread() tells OpenCV to load the image in grayscale
grey_img = cv2.imread("cat.jpg", 0)

hist = cv2.calcHist([grey_img], [0], None, [256], [0, 256])

# cv2.imshow("grey",grey_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plot greyscale histogram
plt.plot(hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Color histogram 
img = cv2.imread("cat.jpg")

# initialize colors for BGR channels
colors = ('b', 'g', 'r')

# loop through each color channel
for i, col in enumerate(colors):
    # calculate histogram for each channel
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()




