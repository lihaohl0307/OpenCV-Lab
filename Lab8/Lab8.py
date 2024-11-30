import cv2

# Sobel Edge Detection

# Load an image
image = cv2.imread('image.jpg', 0)
# Apply Sobel filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # Vertical edges
# Combine Sobel X and Y
sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
# Display result
cv2.imshow("Sobel Edge Detection", sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Canny Edge Detection

# # Load an image
image = cv2.imread('image.jpg', 0)
# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)
# Display result
cv2.imshow("Canny Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()