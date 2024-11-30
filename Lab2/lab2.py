import cv2
import numpy as np

# Load original image
img = cv2.imread("cat.jpg")

# Convert to grayscale
grey_img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image
resized_img = cv2.resize(img, (400, 400))

# Convert to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Add text to the image
text_img = cv2.putText(img.copy(), "SPRITE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)



# Harris Corner Detection
# Make a copy of the original image for Harris corner detection
img_copy = img.copy()

# Convert the copy to grayscale
grey_img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

# Convert to float32 (required by cornerHarris)
grey_img_copy = np.float32(grey_img_copy)

# Apply Harris corner detection
harris_corners = cv2.cornerHarris(grey_img_copy, blockSize=20, ksize=3, k=0.04)

# Dilate the detected corners to enhance them
harris_corners = cv2.dilate(harris_corners, None)

# Threshold for an optimal value to identify strong corners
img_copy[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]




# Display the different images with unique window names
cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", grey_img)
cv2.imshow("Resized Image", resized_img)
cv2.imshow("HSV Image", hsv_image)
cv2.imshow("Text Image", text_img)
cv2.imshow("Harris Corners", img_copy)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grayscale image
cv2.imwrite("cat-grey.jpg", grey_img)
