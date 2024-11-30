import cv2

# Contours

# Load and preprocess the image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# Display result
cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()