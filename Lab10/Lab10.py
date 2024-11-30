import cv2
import numpy as np

# Motion Detection

# Set up camera
cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi Corner Detection to find points for Lucas-Kanade
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Detect initial points to track using Shi-Tomasi Corner Detection
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing Lucas-Kanade optical flow tracks
mask = np.zeros_like(old_frame)

try:
    while True:
        # Capture a new frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if there are points to track
        if p0 is not None and len(p0) > 0:
            # Calculate optical flow using Lucas-Kanade for sparse feature points
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points (those successfully tracked)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks for Lucas-Kanade Optical Flow
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                # Overlay the Lucas-Kanade tracks on the original frame
                lk_output = cv2.add(frame, mask)
                cv2.imshow("Lucas-Kanade Optical Flow (Sparse)", lk_output)

            # Update the previous frame and points for the next loop iteration
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2) if good_new is not None else None
        else:
            # Re-detect points if no good points were found
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        # ---- Dense Optical Flow ----
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        dense_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Dense Optical Flow", dense_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
