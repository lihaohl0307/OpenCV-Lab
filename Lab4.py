import cv2
import numpy as np

#Read the Image
img = cv2.imread("image.jpg")
img_2 = cv2.imread("image2.jpg")
#Resize the Image
resized_img = cv2.resize(img, (500, 500))

#Crop the Image
cropped_img = img[0:200, 200:500]

#Rotate the Image
height, width = img.shape[:2]
center = (width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

#Convert the Image to Grayscale
grey_img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

#Add text to the Image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(grey_img, 'NEU', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#Harries Corner Detection
gray = np.float32(grey_img)
harris_corners = cv2.cornerHarris(gray, blockSize=20, ksize=3, k=0.04)

#Dilate corner image to enhance corner points
harris_corners = cv2.dilate(harris_corners, None)

#Threshold for an optimal value
img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

#Display the Image
cv2.imshow('Harries Corners', img)

#SIFT Detector
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(grey_img, None)
keypoints2, description2 = sift.detectAndCompute(img_2, None)

#FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, description2, k=2)

#Apply Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

#Draw Matches
result = cv2.drawMatches(grey_img, keypoints1, img_2, keypoints2, good_matches, None)

#Draw the Keypoints
#image_with_keypoints = cv2.drawKeypoints(grey_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Show the image with the keypoints
#cv2.imshow('SIFT Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Display the Image
cv2.imshow("Image", grey_img)
cv2.waitKey(0)
#cv2.waitKet(500) #500ms
cv2.destroyAllWindows() #Close the windowimage

#Saving image
cv2.imwrite("lab_2_image.jpg", grey_img)


