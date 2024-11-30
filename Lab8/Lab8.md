## OpenCV Lab Documentation - Edge Detection Techniques

### 1. **Overview**
This lab demonstrates edge detection techniques using OpenCV. Edge detection is a fundamental operation in image processing used to identify boundaries within an image. This documentation covers the Sobel and Canny edge detection methods, explaining their implementation and results.

---

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
```
- **Description**: OpenCV is used for image loading, processing, and display.

---

#### Reading the Image
```python
image = cv2.imread('image.jpg', 0)
```
- **Description**: Reads the input image (`image.jpg`) in grayscale mode.
- **Observation**: Grayscale images simplify edge detection as they focus on intensity differences.

---

### 3. **Edge Detection Techniques**

#### Sobel Edge Detection
1. **Horizontal and Vertical Edge Detection**:
   ```python
   sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # Horizontal edges
   sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # Vertical edges
   ```
   - **Description**: 
     - The `Sobel` function calculates the gradient of the image intensity in the x or y direction.
     - `cv2.CV_64F`: Ensures higher precision for gradient values.
     - `ksize=3`: Specifies the size of the Sobel kernel (filter).
   - **Observation**: 
     - `sobel_x` detects horizontal edges.
     - `sobel_y` detects vertical edges.

2. **Combining Gradients**:
   ```python
   sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
   ```
   - **Description**: Combines the horizontal and vertical gradients using the Euclidean formula to compute the overall edge intensity.
   - **Observation**: Results in a single image highlighting all edges.

3. **Display Result**:
   ```python
   cv2.imshow("Sobel Edge Detection", sobel_combined)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   - **Description**: Displays the result of Sobel edge detection.

---

#### Canny Edge Detection
1. **Apply Canny Edge Detection**:
   ```python
   edges = cv2.Canny(image, 100, 200)
   ```
   - **Description**: 
     - The `Canny` function detects edges using a multi-stage process:
       1. Noise reduction with Gaussian filtering.
       2. Gradient computation.
       3. Non-maximum suppression.
       4. Edge tracing by hysteresis thresholding.
     - `100, 200`: Lower and upper thresholds for edge linking.
   - **Observation**: Produces sharp, well-defined edges.

2. **Display Result**:
   ```python
   cv2.imshow("Canny Edge Detection", edges)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   - **Description**: Displays the result of Canny edge detection.

---

### 4. **Execution and Results**

To run the code, use:
```bash
python3 edge_detection.py
```
- **Expected Output**:
  - Sobel Edge Detection: An image displaying combined horizontal and vertical edges.
  - Canny Edge Detection: An image displaying sharp and well-defined edges.

---

### 5. **Observations**
- **Sobel Edge Detection**:
  - Highlights edges in horizontal and vertical directions.
  - Best for detecting gradients and analyzing edge orientation.
- **Canny Edge Detection**:
  - More robust and accurate for detecting edges.
  - Effective for noisy images due to built-in Gaussian smoothing.

---

### 6. **Conclusion**
This lab highlights the capabilities of Sobel and Canny edge detection:
- Sobel is useful for gradient analysis and detecting edges in specific directions.
- Canny is more sophisticated and ideal for applications requiring precise edge detection.

Both techniques are foundational tools in computer vision, applicable in tasks like object detection, segmentation, and image understanding.
