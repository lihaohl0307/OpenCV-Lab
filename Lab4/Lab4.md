## OpenCV Lab Documentation - Feature Detection and Matching

### 1. **Overview**
This lab demonstrates various image processing techniques using OpenCV, including resizing, cropping, grayscale conversion, text addition, Harris corner detection, SIFT feature detection, and feature matching with FLANN. 

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
```
We use OpenCV for image processing tasks and NumPy for handling numerical data.

#### Reading Images
```python
img = cv2.imread("image.jpg")
img_2 = cv2.imread("image2.jpg")
```
- **Description**: `image.jpg` and `image2.jpg` are loaded for processing. The primary image is `img`, while `img_2` is used for feature matching.

#### Resizing and Cropping the Image
1. **Resize**:
   ```python
   resized_img = cv2.resize(img, (500, 500))
   ```
   - **Description**: Resizes `img` to 500x500 pixels.
   - **Observation**: Resizing standardizes image size, making it easier to visualize and compare across various steps.

2. **Crop**:
   ```python
   cropped_img = img[0:200, 200:500]
   ```
   - **Description**: Crops a specific section of the image.
   - **Observation**: Cropping allows us to focus on a particular region within the image.

#### Rotate the Image
```python
height, width = img.shape[:2]
center = (width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
```
- **Description**: Defines a rotation matrix for rotating the image by 45 degrees.
- **Observation**: This prepares the image for rotated viewing, useful in various visualization tasks.

#### Convert to Grayscale
```python
grey_img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
```
- **Description**: Converts the image to grayscale.
- **Observation**: Grayscale simplifies the image data, essential for corner and feature detection.

#### Add Text to the Image
```python
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(grey_img, 'NEU', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
```
- **Description**: Adds the text "NEU" to `grey_img`.
- **Observation**: Text annotations are helpful for labeling images, especially in documentation.

#### Harris Corner Detection
```python
gray = np.float32(grey_img)
harris_corners = cv2.cornerHarris(gray, blockSize=20, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)
img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
cv2.imshow('Harries Corners', img)
```
- **Description**: Detects corners in the image, highlighting them with red dots.
- **Observation**: Corner detection is valuable for identifying regions of interest in image analysis.

#### SIFT Feature Detection
```python
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(grey_img, None)
keypoints2, description2 = sift.detectAndCompute(img_2, None)
```
- **Description**: Detects keypoints and computes descriptors for both `grey_img` and `img_2`.
- **Observation**: SIFT (Scale-Invariant Feature Transform) is robust for detecting features across images, regardless of scale and rotation.

#### FLANN-Based Matcher for Feature Matching
```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, description2, k=2)
```
- **Description**: Sets up a FLANN-based matcher to find feature matches between the two images.
- **Observation**: FLANN is efficient for large datasets, making it ideal for feature matching.

#### Apply Ratio Test and Draw Matches
```python
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
result = cv2.drawMatches(grey_img, keypoints1, img_2, keypoints2, good_matches, None)
```
- **Description**: Applies the ratio test to filter out weak matches, then draws the matching keypoints between images.
- **Observation**: The ratio test enhances match quality by filtering out ambiguous matches.

#### Save and Display Images
```python
cv2.imshow("Image", grey_img)
cv2.imwrite("lab_2_image.jpg", grey_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- **Description**: Displays and saves the processed grayscale image.
- **Observation**: Saving processed images aids in further analysis and documentation.

### 3. **Execution and Results**
To run the code, use:
```bash
python3 Lab2.py
```
- **Expected Output**: Various transformations of `image.jpg` are displayed, and feature matches are visualized between `image.jpg` and `image2.jpg`.
- **Note**: Ensure both images are accessible in the specified path.

### 4. **Observations**
- **Harris Corner and SIFT**: Harris corner detection works well for simple edge detection, while SIFT provides a more robust method for feature matching across images.
- **FLANN Matcher**: Useful for comparing large datasets, such as matching features between multiple images.

### 5. **Conclusion**
This lab demonstrates essential OpenCV functionalities for image processing and feature matching, combining grayscale conversion, transformations, Harris corner detection, SIFT, and FLANN-based matching.
