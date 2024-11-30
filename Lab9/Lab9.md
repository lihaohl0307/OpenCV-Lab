## OpenCV Lab Documentation - Contour Detection

### 1. **Overview**
This lab demonstrates how to detect and draw contours in an image using OpenCV. Contours are curves that connect all continuous points along a boundary with the same color or intensity. Contour detection is widely used in shape analysis, object detection, and image segmentation.

---

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
```
- **Description**: OpenCV is used for image loading, preprocessing, and contour detection.

---

#### Reading and Preprocessing the Image
1. **Load the Image**:
   ```python
   image = cv2.imread('image.jpg')
   ```
   - **Description**: Reads the input image (`image.jpg`) in its original color format.

2. **Convert to Grayscale**:
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```
   - **Description**: Converts the image to grayscale, reducing the color space and simplifying contour detection.

3. **Apply Binary Thresholding**:
   ```python
   _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
   ```
   - **Description**: Converts the grayscale image into a binary image. Pixels with intensity greater than 127 are set to 255 (white), and the rest are set to 0 (black).
   - **Observation**: Binary images make it easier to detect contours.

---

#### Detecting and Drawing Contours
1. **Find Contours**:
   ```python
   contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```
   - **Description**:
     - `cv2.RETR_EXTERNAL`: Retrieves only the outermost contours.
     - `cv2.CHAIN_APPROX_SIMPLE`: Compresses horizontal, vertical, and diagonal segments to save memory.
   - **Output**:
     - `contours`: A list of detected contours.
     - `hierarchy`: Hierarchical relationships between contours (if applicable).

2. **Draw Contours**:
   ```python
   cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
   ```
   - **Description**: Draws all detected contours on the original image.
   - **Parameters**:
     - `-1`: Draws all contours.
     - `(0, 255, 0)`: Specifies the color of the contours (green in BGR format).
     - `2`: Specifies the thickness of the contour lines.
   - **Observation**: Detected contours are visually highlighted.

---

#### Displaying the Result
```python
cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
- **Description**:
  - Displays the image with contours drawn.
  - Waits for a key press to close the display window.

---

### 3. **Execution and Results**

To run the code, use:
```bash
python3 contour_detection.py
```
- **Expected Output**:
  - The input image with contours drawn in green.

---

### 4. **Observations**
- **Thresholding**: Accurate preprocessing is essential for effective contour detection.
- **Contours**: Outermost contours are detected, and their shapes are drawn on the original image.

---

### 5. **Conclusion**
This lab demonstrates how to use OpenCV’s `cv2.findContours` to detect and draw contours in an image. Contour detection is a powerful technique in computer vision, enabling applications such as shape analysis, object tracking, and segmentation.
