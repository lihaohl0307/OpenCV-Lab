## OpenCV Lab Documentation - Image Filtering Techniques

### 1. **Overview**
This lab demonstrates different image filtering techniques using OpenCV. Filtering is used for noise reduction, smoothing, and enhancing image quality. Here, we apply Averaging, Gaussian, Median, and Bilateral filters to observe their effects on an image.

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
```
- **Description**: We use OpenCV for image processing, and NumPy for handling numerical operations.

#### Reading and Displaying the Image
```python
img = cv2.imread("image.jpg")
cv2.imshow("img", img)
cv2.waitKey(0)
```
- **Description**: Loads `image.jpg` and displays the original image.
- **Observation**: The original image provides a reference for comparing the effects of each filter.

#### Applying Different Filters
1. **Averaging (Box Filter)**:
   ```python
   blurred_avg = cv2.blur(img, (5,5))
   cv2.imshow("Averaging", blurred_avg)
   cv2.waitKey(0)
   ```
   - **Description**: Uses a 5x5 kernel to apply a basic averaging filter, which smoothens the image by averaging pixel values in the kernel.
   - **Observation**: Averaging blurs the image evenly but does not preserve edges well.

2. **Gaussian Filter**:
   ```python
   blurred_gaussian = cv2.GaussianBlur(img, (5,5), 0)
   cv2.imshow("Gaussian", blurred_gaussian)
   cv2.waitKey(0)
   ```
   - **Description**: Applies a Gaussian filter with a 5x5 kernel. Gaussian filtering gives more weight to pixels near the center of the kernel, producing a smoother effect.
   - **Observation**: Gaussian filtering is effective for noise reduction while slightly preserving edges.

3. **Median Filter**:
   ```python
   blurred_median = cv2.medianBlur(img, 5)
   cv2.imshow("Median", blurred_median)
   cv2.waitKey(0)
   ```
   - **Description**: Applies a median filter with a 5x5 kernel, replacing each pixel with the median of its neighbors.
   - **Observation**: Median filtering is effective for reducing "salt and pepper" noise while preserving edges well.

4. **Bilateral Filter**:
   ```python
   blurred_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
   cv2.imshow("Bilateral", blurred_bilateral)
   cv2.waitKey(0)
   ```
   - **Description**: Applies a bilateral filter with a diameter of 9 and two sigma values for color and space. Bilateral filtering smooths regions while preserving edges.
   - **Observation**: This filter effectively reduces noise without blurring edges, making it ideal for facial images or scenarios where edge preservation is critical.

#### Saving the Filtered Images
```python
cv2.imwrite("Averaging.jpg", blurred_avg)
cv2.imwrite("Gaussian.jpg", blurred_gaussian)
cv2.imwrite("Median.jpg", blurred_median)
cv2.imwrite("Bilateral.jpg", blurred_bilateral)
```
- **Description**: Saves each filtered image for future reference or comparison.
- **Observation**: Saving processed images helps with analysis and documentation.

#### Closing All Windows
```python
cv2.destroyAllWindows()
```
- **Description**: Closes all OpenCV windows after displaying images.
- **Observation**: Important to release system resources used for displaying images.

### 3. **Execution and Results**
To run the code, use:
```bash
python3 Lab_Filters.py
```
- **Expected Output**: Displays the original image followed by four filtered versions using different techniques.
- **Note**: Ensure `image.jpg` is accessible in the same directory as the code file.

### 4. **Observations**
- **Averaging Filter**: Basic smoothing but lacks edge preservation.
- **Gaussian Filter**: Smooths and reduces noise with some edge preservation.
- **Median Filter**: Effective for removing specific noise types and preserving edges.
- **Bilateral Filter**: Maintains edges while smoothing regions, ideal for complex images.

### 5. **Conclusion**
This lab illustrates the effect of different filters, each suited for specific image processing tasks. Bilateral filtering is generally preferred when edge preservation is important, while median filtering is ideal for noise reduction.

