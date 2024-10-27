import cv2
import numpy as np

#Read the Image
img = cv2.imread("image.jpg")
cv2.imshow("img", img)
cv2.waitKey(0)

#Apply Averaging(Box Filter)
blurred_avg = cv2.blur(img, (5,5))
#Display the blurred image
cv2.show("Averaging", blurred_avg)
cv2.waitKey(0)

#Apply Gaussian Filter
blurred_gaussian = cv2.GaussianBlur(img, (5,5), 0)
#Display the blurred image
cv2.show("Gaussian", blurred_gaussian)
cv2.waitKey(0)

#Apply Median Filter
blurred_median = cv2.medianBlur(img, 5)
#Display the blurred image
cv2.show("Median", blurred_median)
cv2.waitKey(0)

#Apply Bilateral Filter
blurred_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
#Display the blurred image
cv2.show("Bilateral", blurred_bilateral)
cv2.waitKey(0)

#Save all the images
cv2.imwrite("Averaging.jpg", blurred_avg)
cv2.imwrite("Gaussian.jpg", blurred_gaussian)
cv2.imwrite("Median.jpg", blurred_median)
cv2.imwrite("Bilateral.jpg", blurred_bilateral)

#Close all the windows
cv2.destroyAllWindows()
