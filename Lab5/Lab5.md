## OpenCV Lab Documentation - Grayscale and Color Histograms

### 1. **Overview**
This lab demonstrates how to use OpenCV to calculate and display grayscale and color histograms of an image. Histograms provide insights into the distribution of pixel intensities across different color channels, which is essential for understanding image characteristics and performing adjustments like contrast enhancement.

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
- **Description**: We import OpenCV for image processing, NumPy for handling data structures, and Matplotlib for plotting histograms.

#### Loading the Image in Grayscale
```python
grey_img = cv2.imread("cat.jpg", 0)
```
- **Description**: Loads `cat.jpg` in grayscale mode by using `0` in `cv2.imread()`.
- **Observation**: Grayscale reduces the image to one channel, simplifying intensity analysis.

#### Calculating and Plotting the Grayscale Histogram
```python
hist = cv2.calcHist([grey_img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
```
- **Description**: Calculates and plots the histogram for grayscale intensities.
  - `cv2.calcHist([grey_img], [0], None, [256], [0, 256])`: Calculates the frequency of each pixel intensity (0-255) in the grayscale image.
- **Observation**: The histogram shows the distribution of brightness levels, with peaks indicating frequently occurring intensities.

#### Loading the Image in Color
```python
img = cv2.imread("cat.jpg")
```
- **Description**: Loads `cat.jpg` in color mode (default).
- **Observation**: A color image has three channels (Blue, Green, Red), each representing different intensity distributions.

#### Calculating and Plotting the Color Histogram
```python
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
```
- **Description**: Loops through each color channel (B, G, R) to calculate and plot its histogram:
  - `cv2.calcHist([img], [i], None, [256], [0, 256])`: Calculates the frequency of pixel intensities for each color channel.
  - `plt.plot(hist, color=col)`: Plots each channel’s histogram in its respective color.
- **Observation**: Separate histograms for each channel reveal intensity variations in Blue, Green, and Red, which can be useful for understanding and adjusting color balance.

### 3. **Execution and Results**
To run the code, use:
```bash
python3 Lab_Histogram.py
```
- **Expected Output**: Two plots—one grayscale histogram and one color histogram (showing three separate color channels).
- **Note**: Ensure `cat.jpg` is located in the same directory as the code file.

### 4. **Observations**
- **Grayscale Histogram**: Provides a single-channel intensity distribution, highlighting how brightness is spread throughout the image.
- **Color Histogram**: Shows individual distributions of the B, G, and R channels, useful for tasks like color correction and filtering.

### 5. **Conclusion**
This lab demonstrates how histograms provide a statistical view of image intensities. Grayscale histograms are helpful for brightness and contrast analysis, while color histograms aid in understanding and adjusting color balance.

