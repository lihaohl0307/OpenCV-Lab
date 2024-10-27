import cv2
import numpy as np

#Loading Images
img = cv2.imread("image.jpg")

#Loads the image in greyscale
grey_img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

#Resizing an image
resized_img = cv2.resize(img, (500, 500))

#Cropping an image
cropped_img = img[80:280, 150:330]

#Rotating an image
height, width = img.shape[:2]
center = (width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

#Drawing a line on an image
line_img = cv2.line(img, (250, 500), (5000, 5000), (0, 0, 0), 20)
cv2.imshow("Line Image", line_img)

#Drawing a rectangle on an image
rectangle_img = cv2.rectangle(img, (250, 250), (500, 500), (0, 0, 0), 20)
cv2.imshow("Rectangle Image", rectangle_img)

#Drawing a circle on an image
circle_img = cv2.circle(img, (400, 400), 100, (0, 0, 0), 20)
cv2.imshow("Circle Image", circle_img)

#Adding text to an image
font = cv2.FONT_HERSHEY_SIMPLEX
text_img = cv2.putText(img, 'NEU', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Text Image", text_img)


#Displaying images
cv2.imshow("resized Image", resized_img)
cv2.imshow("cropped Image", cropped_img)
cv2.imshow("rotated Image", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
