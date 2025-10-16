'''
Simple object tracker using IoU-based matching and Kalman filter
Similar to SORT (Simple Online and Realtime Tracking)
'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class Track:
    """Single object track with Kalman filter"""
    count = 0

    def __init__(self, bbox, cls_idx):
        """
        bbox: [x, y, z, l, w, h, theta]
        """
        self.id = Track.count
        Track.count += 1

        self.cls_idx = cls_idx
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # Initialize Kalman filter (state: x, y, z, vx, vy, vz)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1],  # z = z + vz
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1],  # vz = vz
        ])

        # Measurement matrix (we measure x, y, z)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R *= 1.0

        # Process noise
        self.kf.Q[-3:, -3:] *= 0.01

        # Initial state
        self.kf.x[:3] = bbox[:3].reshape(3, 1)

        # Store bbox dimensions
        self.l, self.w, self.h = bbox[3:6]
        self.theta = bbox[6]

    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        """Update with new detection"""
        self.time_since_update = 0
        self.hits += 1

        # Update position
        self.kf.update(bbox[:3])

        # Update dimensions (exponential moving average)
        alpha = 0.3
        self.l = alpha * bbox[3] + (1 - alpha) * self.l
        self.w = alpha * bbox[4] + (1 - alpha) * self.w
        self.h = alpha * bbox[5] + (1 - alpha) * self.h
        self.theta = bbox[6]  # Just use latest angle

    def get_state(self):
        """Get current state as bbox [x, y, z, l, w, h, theta]"""
        return np.array([
            self.kf.x[0, 0],
            self.kf.x[1, 0],
            self.kf.x[2, 0],
            self.l,
            self.w,
            self.h,
            self.theta
        ])

def iou_2d(bbox1, bbox2):
    """
    Compute 2D IoU for bird's eye view
    bbox: [x, y, z, l, w, h, theta]
    Simplified: use axis-aligned bounding boxes
    """
    x1, y1, l1, w1 = bbox1[0], bbox1[1], bbox1[3], bbox1[4]
    x2, y2, l2, w2 = bbox2[0], bbox2[1], bbox2[3], bbox2[4]

    # Get bounding box corners (axis-aligned)
    box1_x1, box1_x2 = x1 - l1/2, x1 + l1/2
    box1_y1, box1_y2 = y1 - w1/2, y1 + w1/2

    box2_x1, box2_x2 = x2 - l2/2, x2 + l2/2
    box2_y1, box2_y2 = y2 - w2/2, y2 + w2/2

    # Compute intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y1 = max(box1_y1, box2_y1)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Compute union
    box1_area = l1 * w1
    box2_area = l2 * w2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

class SimpleTracker:
    """Multi-object tracker"""

    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        """
        max_age: maximum frames to keep track without update
        min_hits: minimum hits before track is confirmed
        iou_threshold: minimum IoU for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections, scores, labels):
        """
        Update tracks with new detections

        Args:
            detections: Nx7 array [x, y, z, l, w, h, theta]
            scores: N array of confidence scores
            labels: N array of class labels

        Returns:
            List of active tracks as (track_id, bbox, cls_idx, score)
        """
        # Predict new locations
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate(detections, labels)
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))

        # Update matched tracks
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = Track(detections[det_idx], labels[det_idx])
            self.tracks.append(track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                results.append((track.id, track.get_state(), track.cls_idx, 1.0))

        return results

    def _associate(self, detections, labels):
        """Associate detections to tracks using IoU"""
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                # Only match same class
                if labels[d] == track.cls_idx:
                    iou_matrix[d, t] = iou_2d(det, track.get_state())

        # Hungarian algorithm for assignment
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))

        # Filter low IoU matches
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue
            matches.append(m)
        matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        unmatched_detections = [d for d in range(len(detections)) if d not in matches[:, 0]]
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in matches[:, 1]]

        return matches, unmatched_detections, unmatched_tracks
