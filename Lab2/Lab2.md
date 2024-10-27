## OpenCV Image Processing Lab Documentation

### 1. **Overview**
This lab involves processing an image using OpenCV. Key tasks include:
- Loading and resizing images.
- Converting images to grayscale and HSV color spaces.
- Adding text to images.
- Detecting corners using the Harris Corner Detection algorithm.

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
```
We import OpenCV for image processing and NumPy for handling numerical data.

#### Load and Display the Original Image
```python
img = cv2.imread("cat.jpg")
cv2.imshow("Original Image", img)
```
- **Description**: Reads `cat.jpg` and displays the original image.
- **Observation**: Ensure that `cat.jpg` is in the correct directory, or provide the full file path.

#### Convert to Grayscale
```python
grey_img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale Image", grey_img)
```
- **Description**: Loads `cat.jpg` directly as a grayscale image.
- **Observation**: Grayscale images are used for certain image processing tasks, such as edge or corner detection, due to reduced color data.

#### Resize Image
```python
resized_img = cv2.resize(img, (400, 400))
cv2.imshow("Resized Image", resized_img)
```
- **Description**: Resizes the image to 400x400 pixels.
- **Observation**: Resizing may be necessary to fit within a specific layout or reduce processing time.

#### Convert to HSV Color Space
```python
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv_image)
```
- **Description**: Converts the image from BGR (OpenCV default) to HSV color space.
- **Observation**: HSV allows for more robust color detection and manipulation.

#### Add Text to the Image
```python
text_img = cv2.putText(img.copy(), "SPRITE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imshow("Text Image", text_img)
```
- **Description**: Adds the text "SPRITE" to a copy of the original image.
- **Observation**: Text overlays are useful for labeling images.

#### Harris Corner Detection
```python
img_copy = img.copy()
grey_img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
grey_img_copy = np.float32(grey_img_copy)
harris_corners = cv2.cornerHarris(grey_img_copy, blockSize=20, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)
img_copy[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
cv2.imshow("Harris Corners", img_copy)
```
- **Description**: Harris Corner Detection highlights corners in the image by checking for sudden changes in intensity.
- **Observation**: Red dots appear at corners or points of high intensity gradients.

#### Save Grayscale Image
```python
cv2.imwrite("cat-grey.jpg", grey_img)
```
- **Description**: Saves the grayscale version of the image as `cat-grey.jpg`.

### 3. **Execution and Results**
To run the code, use a terminal and ensure OpenCV is installed. Run the script:
```bash
python3 Lab3.py
```
- **Expected Output**: Six images should appear, displaying the original, grayscale, resized, HSV, text overlay, and corner-detected images.
- **Note**: Ensure `cat.jpg` is accessible in the specified path.

### 4. **Observations**
- **Grayscale**: Useful for computational tasks by reducing image complexity.
- **HSV Color Space**: More effective for color-based detection than BGR.
- **Harris Corner Detection**: Highlights areas with significant detail, aiding in feature detection for applications like object recognition.
