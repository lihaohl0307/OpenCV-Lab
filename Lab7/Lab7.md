## OpenCV Lab Documentation - Image Thresholding Techniques

### 1. **Overview**
This lab demonstrates various image thresholding techniques using OpenCV. Thresholding is a key image segmentation technique that simplifies an image by converting it into a binary format based on intensity levels. This documentation explains Simple Thresholding, Adaptive Thresholding (Mean and Gaussian), and Otsu’s Binarization.

---

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
- **Description**: 
  - `cv2` is used for image processing.
  - `numpy` is used for numerical operations (though not used directly in this script).
  - `matplotlib` is used for visualizing the results.

---

#### Reading the Image
```python
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
```
- **Description**: Reads the input image (`image.jpg`) in grayscale mode.
- **Observation**: Grayscale images are required for thresholding operations.

---

#### Simple Thresholding
```python
_, thresh_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh_simple, cmap='gray')
plt.title('Simple Thresholding')
plt.show()
```
- **Description**: Pixels with values greater than 127 are set to 255 (white), and others are set to 0 (black).
- **Observation**: This method provides a global threshold but may not handle varying lighting conditions well.

---

#### Adaptive Thresholding
1. **Mean Adaptive Thresholding**:
   ```python
   thresh_adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
   ```
   - **Description**: Calculates the threshold as the mean intensity of the neighborhood minus a constant (2).
   - **Observation**: Suitable for images with uneven lighting.

2. **Gaussian Adaptive Thresholding**:
   ```python
   thresh_adaptive_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
   ```
   - **Description**: Similar to mean adaptive thresholding but uses a Gaussian-weighted sum of the neighborhood.
   - **Observation**: Provides a smoother result, especially near edges.

```python
plt.subplot(1, 2, 1), plt.imshow(thresh_adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')

plt.subplot(1, 2, 2), plt.imshow(thresh_adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.show()
```
- **Description**: Displays the results of adaptive thresholding techniques side-by-side.

---

#### Otsu’s Binarization
```python
_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh_otsu, cmap='gray')
plt.title('Otsu’s Binarization')
plt.show()
```
- **Description**: Automatically determines the optimal threshold value to minimize intra-class variance.
- **Observation**: Effective for bimodal histograms where foreground and background intensity distributions are distinct.

---

### 3. **Execution and Results**

To run the code, use:
```bash
python3 thresholding_lab.py
```
- **Expected Output**:
  - The original image, followed by visualizations for each thresholding technique.
  - Separate plots for Simple Thresholding, Adaptive Thresholding (Mean and Gaussian), and Otsu’s Binarization.

---

### 4. **Observations**
- **Simple Thresholding**: Works well for uniformly lit images but struggles with uneven lighting.
- **Adaptive Mean Thresholding**: Dynamically calculates thresholds, effectively handling variations in lighting.
- **Adaptive Gaussian Thresholding**: Provides a smoother segmentation with edge preservation.
- **Otsu’s Binarization**: Best for images with a clear distinction between foreground and background.

---

### 5. **Conclusion**
This lab highlights the utility of different thresholding methods in image segmentation. While simple thresholding is easy to implement, adaptive and Otsu's methods are better suited for images with varying illumination or complex intensity distributions.

