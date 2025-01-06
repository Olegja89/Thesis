import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self, smoothing_window=30):
        self.smoothing_window = smoothing_window
        self.prev_gray = None
        self.transforms = []
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def _detect_features(self, frame_gray):
        return cv2.goodFeaturesToTrack(frame_gray, **self.feature_params)

    def _compute_transform(self, prev_pts, curr_pts):
        transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return np.array([dx, dy, da])

    def _smooth_transforms(self):
        if len(self.transforms) < self.smoothing_window:
            return self.transforms[-1]
        
        pad_left = (self.smoothing_window - 1) // 2
        pad_right = self.smoothing_window - pad_left - 1
        smooth = np.convolve(self.transforms[:, 0], np.ones(self.smoothing_window) / self.smoothing_window, mode='valid')
        smooth_dx = smooth[-1]
        smooth = np.convolve(self.transforms[:, 1], np.ones(self.smoothing_window) / self.smoothing_window, mode='valid')
        smooth_dy = smooth[-1]
        smooth = np.convolve(self.transforms[:, 2], np.ones(self.smoothing_window) / self.smoothing_window, mode='valid')
        smooth_da = smooth[-1]
        
        return np.array([smooth_dx, smooth_dy, smooth_da])

    def stabilize_frame(self, frame):
        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame
        
        # Detect feature points in previous frame
        prev_pts = self._detect_features(self.prev_gray)
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = curr_gray
            return frame
        
        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        
        # Filter out bad points
        mask = status.ravel() == 1
        if not np.any(mask):
            self.prev_gray = curr_gray
            return frame
            
        prev_pts = prev_pts[mask]
        curr_pts = curr_pts[mask]
        
        # Calculate transformation
        transform = self._compute_transform(prev_pts, curr_pts)
        self.transforms.append(transform)
        
        # Smooth transforms
        smooth_transform = self._smooth_transforms()
        
        # Apply smoothed transformation
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(smooth_transform[2]), 1.0)
        rotation_matrix[0, 2] += smooth_transform[0]
        rotation_matrix[1, 2] += smooth_transform[1]
        
        stabilized_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        
        # Update previous frame
        self.prev_gray = curr_gray
        
        return stabilized_frame