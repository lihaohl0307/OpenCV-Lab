## OpenCV Lab Documentation - Motion Detection using Optical Flow

### 1. **Overview**
This lab demonstrates motion detection using two types of optical flow techniques in OpenCV:
1. **Sparse Optical Flow**: Tracks specific feature points using the Lucas-Kanade method.
2. **Dense Optical Flow**: Visualizes motion across the entire frame using the Farneback algorithm.

These methods are essential for analyzing motion patterns, detecting moving objects, and video stabilization.

---

### 2. **Code Explanation**

#### Import Libraries
```python
import cv2
import numpy as np
```
- **Description**: 
  - OpenCV is used for real-time video capture and motion analysis.
  - NumPy handles numerical operations for optical flow visualization.

---

#### Camera Setup
```python
cap = cv2.VideoCapture(0)
```
- **Description**: Initializes the webcam feed for capturing frames.
- **Observation**: Ensure your webcam is accessible for this to work.

---

#### Sparse Optical Flow (Lucas-Kanade)
1. **Shi-Tomasi Corner Detection**:
   ```python
   feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
   ```
   - **Description**: Detects up to 100 good feature points for tracking based on corner detection criteria.

2. **Lucas-Kanade Optical Flow Parameters**:
   ```python
   lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
   ```
   - **Description**:
     - `winSize`: Size of the window used for tracking.
     - `maxLevel`: Pyramid levels for tracking.
     - `criteria`: Stopping criteria for the iterative tracking process.

3. **Tracking Points**:
   ```python
   p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
   ```
   - **Description**: Detects good features to track in the initial frame.

4. **Calculate and Visualize Tracks**:
   ```python
   p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
   ```
   - **Description**: Computes the optical flow for detected feature points between consecutive frames.
   - **Visualization**: Tracks are drawn using green lines and red circles for tracked points.

---

#### Dense Optical Flow (Farneback)
1. **Calculate Dense Optical Flow**:
   ```python
   flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
   ```
   - **Description**:
     - Computes dense motion vectors for all pixels in the frame.
     - Parameters control pyramid scale, iterations, and smoothing.

2. **Visualize Dense Motion**:
   ```python
   mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
   hsv[..., 0] = ang * 180 / np.pi / 2
   hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
   dense_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
   ```
   - **Description**: Motion is visualized in HSV space:
     - Hue: Represents motion direction.
     - Value: Represents motion magnitude.

---

### 3. **Execution and Results**

To run the code, use:
```bash
python3 motion_detection.py
```
- **Expected Output**:
  - **Sparse Optical Flow**: Displays tracked motion as green lines connecting tracked points with red circles for feature points.
  - **Dense Optical Flow**: Displays motion as a color map where hue indicates direction and intensity indicates magnitude.

Press `q` to exit the program.

---

### 4. **Observations**
- **Sparse Optical Flow**:
  - Suitable for tracking a small number of features.
  - Computationally efficient and effective for applications like object tracking.
- **Dense Optical Flow**:
  - Provides a comprehensive view of motion across the entire frame.
  - Computationally expensive but useful for analyzing global motion patterns.

---

### 5. **Conclusion**
This lab demonstrates the implementation of motion detection using optical flow techniques:
- **Lucas-Kanade Sparse Optical Flow**: Ideal for feature-specific motion tracking.
- **Farneback Dense Optical Flow**: Suitable for visualizing motion patterns across the entire frame.

Both techniques are widely used in video analysis, autonomous vehicles, and motion estimation applications.
