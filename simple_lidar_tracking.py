"""
Simple LiDAR tracking using Hungarian algorithm + Kalman filter
Works in env_py311 - no dependency issues!
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.animation as animation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Paths
dataroot = '/mnt/ssd1/divake/nuScenes_tracking_project/datasets'
prediction_file = '/mnt/ssd1/divake/nuScenes_tracking_project/repos/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/prediction.pkl'
output_dir = '/mnt/ssd1/divake/nuScenes_tracking_project/visualizations/detection_videos'
os.makedirs(output_dir, exist_ok=True)

# Load nuScenes
print("Loading nuScenes...")
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

# Load predictions
print("Loading CenterPoint predictions...")
with open(prediction_file, 'rb') as f:
    predictions = pickle.load(f)

# Get scene_1
scene = nusc.scene[1]
sample_token = scene['first_sample_token']
samples = []
while sample_token:
    sample = nusc.get('sample', sample_token)
    samples.append(sample)
    sample_token = sample['next']

print(f"Scene: {scene['name']}")
print(f"Frames: {len(samples)}\n")

# Simple 3D Kalman Filter Tracker
class SimpleTracker:
    def __init__(self, detection, track_id):
        """Initialize tracker with first detection"""
        self.id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # State: [x, y, z, vx, vy, vz, w, l, h]
        self.kf = KalmanFilter(dim_x=9, dim_z=6)

        # State transition matrix (constant velocity model)
        dt = 0.5  # ~0.5s between frames
        self.kf.F = np.eye(9)
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

        # Measurement matrix (we observe x,y,z,w,l,h)
        self.kf.H = np.zeros((6, 9))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 6] = 1  # w
        self.kf.H[4, 7] = 1  # l
        self.kf.H[5, 8] = 1  # h

        # Initialize state
        x, y, z, w, l, h, yaw = detection
        self.kf.x = np.array([x, y, z, 0, 0, 0, w, l, h])
        self.yaw = yaw

        # Covariance matrices
        self.kf.R *= 0.5  # Measurement noise
        self.kf.P *= 10   # Initial uncertainty
        self.kf.Q *= 0.1  # Process noise

    def update(self, detection):
        """Update tracker with matched detection"""
        self.time_since_update = 0
        self.hits += 1
        x, y, z, w, l, h, yaw = detection
        self.kf.update(np.array([x, y, z, w, l, h]))
        self.yaw = yaw  # Simple yaw update

    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        """Get current state as [x, y, z, w, l, h, yaw]"""
        return np.array([
            self.kf.x[0], self.kf.x[1], self.kf.x[2],
            self.kf.x[6], self.kf.x[7], self.kf.x[8],
            self.yaw
        ])

def iou_3d_simple(box1, box2):
    """Simplified 3D IoU using 2D bird's eye view"""
    x1, y1, z1, w1, l1, h1, yaw1 = box1
    x2, y2, z2, w2, l2, h2, yaw2 = box2

    # 2D distance-based similarity (simpler than true IoU)
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    size_sim = min(l1, l2) * min(w1, w2) / (max(l1, l2) * max(w1, w2) + 1e-6)

    # Combined score (higher is better match)
    score = size_sim / (1 + dist)
    return score

# Tracking parameters
MAX_AGE = 3
MIN_HITS = 1
IOU_THRESHOLD = 0.1

# Process frames
trackers = []
next_id = 1
all_frames_data = []

log_path = os.path.join(output_dir, 'simple_tracking_log.txt')

with open(log_path, 'w') as log:
    log.write(f"Simple Kalman Filter Tracking - Scene {scene['name']}\n")
    log.write(f"{'='*80}\n\n")

    for frame_idx, sample in enumerate(samples):
        print(f"\rProcessing frame {frame_idx+1}/{len(samples)}", end='')

        # Get predictions
        if sample['token'] not in predictions:
            continue

        pred_data = predictions[sample['token']]

        # Get LiDAR points
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])
        pcl = LidarPointCloud.from_file(lidar_filepath)
        points = pcl.points[:3, :].T

        # Extract car detections
        boxes = pred_data['box3d_lidar']
        scores = pred_data['scores']
        labels = pred_data['label_preds']

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 0 and score > 0.3:  # Cars only, score > 0.3
                # [x, y, z, dx, dy, dz, yaw] -> [x, y, z, w, l, h, yaw]
                detections.append([box[0], box[1], box[2], box[4], box[3], box[5], box[6]])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 7))

        # Predict existing trackers
        for trk in trackers:
            trk.predict()

        # Match detections to trackers
        if len(trackers) > 0 and len(detections) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(detections), len(trackers)))
            for d, det in enumerate(detections):
                for t, trk in enumerate(trackers):
                    cost_matrix[d, t] = -iou_3d_simple(det, trk.get_state())

            # Hungarian algorithm
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)

            # Update matched trackers
            matched_dets = set()
            matched_trks = set()

            for d_idx, t_idx in zip(det_indices, trk_indices):
                if -cost_matrix[d_idx, t_idx] >= IOU_THRESHOLD:
                    trackers[t_idx].update(detections[d_idx])
                    matched_dets.add(d_idx)
                    matched_trks.add(t_idx)

            # Create new trackers for unmatched detections
            for d_idx in range(len(detections)):
                if d_idx not in matched_dets:
                    trackers.append(SimpleTracker(detections[d_idx], next_id))
                    next_id += 1

        elif len(detections) > 0:
            # No existing trackers, create new ones
            for det in detections:
                trackers.append(SimpleTracker(det, next_id))
                next_id += 1

        # Remove dead trackers
        trackers = [trk for trk in trackers if trk.time_since_update < MAX_AGE]

        # Get confirmed tracks
        tracked_objects = []
        for trk in trackers:
            if trk.hits >= MIN_HITS:
                state = trk.get_state()
                tracked_objects.append({
                    'box': state,
                    'track_id': trk.id,
                })

        # Log
        log.write(f"Frame {frame_idx+1}/{len(samples)}\n")
        log.write(f"  Detections: {len(detections)}\n")
        log.write(f"  Active trackers: {len(trackers)}\n")
        log.write(f"  Confirmed tracks: {len(tracked_objects)}\n")
        for obj in tracked_objects:
            state = obj['box']
            log.write(f"    ID {obj['track_id']:3d}: pos=({state[0]:6.2f}, {state[1]:6.2f}, {state[2]:5.2f})\n")
        log.write("\n")

        # Store data
        all_frames_data.append({
            'points': points,
            'tracked_objects': tracked_objects,
            'frame_idx': frame_idx,
        })

print(f"\n\n{'='*80}")
print(f"Tracking complete! {len(all_frames_data)} frames processed")
print(f"Log: {log_path}")
print(f"{'='*80}\n")

# Visualization
print("Generating tracking visualization...")

track_trajectories = {}

def get_color_for_id(track_id):
    np.random.seed(track_id)
    return plt.cm.tab20(track_id % 20)

def get_corners(box):
    """Get 2D corners from box [x, y, z, w, l, h, yaw]"""
    x, y, z, w, l, h, yaw = box
    corners_x = np.array([l/2, l/2, -l/2, -l/2])
    corners_y = np.array([w/2, -w/2, -w/2, w/2])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    corners_x_rot = corners_x * cos_yaw - corners_y * sin_yaw + x
    corners_y_rot = corners_x * sin_yaw + corners_y * cos_yaw + y

    return np.stack([corners_x_rot, corners_y_rot], axis=1)

fig, ax = plt.subplots(figsize=(14, 14))

def update_frame(frame_idx):
    ax.clear()

    frame_data = all_frames_data[frame_idx]
    points = frame_data['points']
    tracked_objects = frame_data['tracked_objects']

    # LiDAR points
    ax.scatter(points[:, 1], points[:, 0], c='lightblue', s=0.5, alpha=0.3)

    # Tracked objects
    for obj in tracked_objects:
        box = obj['box']
        track_id = obj['track_id']
        x, y = box[0], box[1]

        # Update trajectory
        if track_id not in track_trajectories:
            track_trajectories[track_id] = []
        track_trajectories[track_id].append((x, y))

        # Plot trajectory
        if len(track_trajectories[track_id]) > 1:
            traj = np.array(track_trajectories[track_id][-10:])
            color = get_color_for_id(track_id)
            ax.plot(traj[:, 1], traj[:, 0], 'o-', color=color, alpha=0.5, linewidth=2, markersize=3)

        # Bounding box
        corners = get_corners(box)
        corners_swapped = corners[:, [1, 0]]

        color = get_color_for_id(track_id)
        poly = Polygon(corners_swapped, fill=False, edgecolor=color, linewidth=3, alpha=1.0)
        ax.add_patch(poly)

        # Track ID
        ax.text(y, x, f"ID {track_id}", fontsize=13, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.95),
               color='white', fontweight='bold')

    # Ego vehicle
    ego_box = Rectangle((-1, -1.5), 2, 3, linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.6)
    ax.add_patch(ego_box)

    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_xlabel('Y - Left/Right (meters)', fontsize=13)
    ax.set_ylabel('X - Forward (meters)', fontsize=13)
    ax.set_title(f'Simple Kalman Filter Tracking - Frame {frame_idx+1}/{len(all_frames_data)}\n'
                f'Tracked Cars: {len(tracked_objects)} | Total Tracks: {len(track_trajectories)}',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

anim = animation.FuncAnimation(fig, update_frame, frames=len(all_frames_data),
                              interval=500, repeat=True)

output_path = os.path.join(output_dir, 'tracking_scene_1_simple_kalman.gif')
print(f"Saving to: {output_path}")
anim.save(output_path, writer='pillow', fps=2, dpi=100)
plt.close()

print(f"\n{'='*80}")
print(f"✓ Tracking GIF: {output_path}")
print(f"✓ Log file: {log_path}")
print(f"{'='*80}\n")
print("Done!")
