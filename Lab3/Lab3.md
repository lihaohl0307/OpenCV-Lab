## OpenCV Image Processing Lab Documentation

### 1. **Overview**
This lab demonstrates several image processing operations using OpenCV, such as loading, resizing, cropping, rotating images, and drawing shapes and text. Each step includes code explanations and observations.

### 2. **Code Explanation**

#### Importing Libraries
```python
import cv2
import numpy as np
```
We import OpenCV for image manipulation and NumPy for numerical operations.

#### Loading the Image
```python
img = cv2.imread("image.jpg")
grey_img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
```
- **Description**: `image.jpg` is loaded in both color and grayscale modes.
- **Observation**: Grayscale is often used for feature extraction and reduces computational complexity.

#### Resizing the Image
```python
resized_img = cv2.resize(img, (500, 500))
cv2.imshow("resized Image", resized_img)
```
- **Description**: Resizes the image to 500x500 pixels.
- **Observation**: This can help with standardizing image size for further processing steps.

#### Cropping the Image
```python
cropped_img = img[80:280, 150:330]
cv2.imshow("cropped Image", cropped_img)
```
- **Description**: Crops a region from (80, 150) to (280, 330).
- **Observation**: Cropping isolates a specific area of interest.

#### Rotating the Image
```python
height, width = img.shape[:2]
center = (width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("rotated Image", rotated_img)
```
- **Description**: Rotates the image by 45 degrees and later by 90 degrees clockwise.
- **Observation**: Rotation can aid in perspective correction or creating visual effects.

#### Drawing Shapes
1. **Drawing a Line**
   ```python
   line_img = cv2.line(img.copy(), (250, 500), (500, 500), (0, 0, 0), 20)
   cv2.imshow("Line Image", line_img)
   ```
   - **Description**: Draws a black line on the image from point (250, 500) to (500, 500).
   - **Observation**: Line thickness and color can be customized, useful for markup.

2. **Drawing a Rectangle**
   ```python
   rectangle_img = cv2.rectangle(img.copy(), (250, 250), (500, 500), (0, 0, 0), 20)
   cv2.imshow("Rectangle Image", rectangle_img)
   ```
   - **Description**: Draws a black rectangle with a 20-pixel thickness.
   - **Observation**: Rectangles are useful for highlighting regions in an image.

3. **Drawing a Circle**
   ```python
   circle_img = cv2.circle(img.copy(), (400, 400), 100, (0, 0, 0), 20)
   cv2.imshow("Circle Image", circle_img)
   ```
   - **Description**: Draws a circle with a radius of 100 pixels and black color.
   - **Observation**: Circles can indicate focal points.

#### Adding Text
```python
font = cv2.FONT_HERSHEY_SIMPLEX
text_img = cv2.putText(img.copy(), 'NEU', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Text Image", text_img)
```
- **Description**: Adds the text "NEU" to the image in green.
- **Observation**: Text overlays are essential for labeling or annotating images.

### 3. **Execution and Results**
To run the code, use:
```bash
python3 Lab3.py
```
- **Expected Output**: Displays various processed images, such as resized, cropped, rotated, and shapes drawn on the original image.
- **Note**: Ensure `image.jpg` is in the same directory as the code file.

### 4. **Observations**
- **Shapes and Text**: Useful for annotating or highlighting areas.
- **Resizing and Cropping**: Standardizes image dimensions, often needed for feature extraction.
- **Rotation**: Adds flexibility for transforming the orientation.

### 5. **Conclusion**
This lab demonstrates foundational OpenCV operations. Each function plays a role in preprocessing images for tasks such as detection, annotation, and feature extraction.

