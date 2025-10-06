#!/usr/bin/env python3
"""
Visualize CenterPoint detections on nuScenes mini dataset and generate video
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.animation as animation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import torch
from PIL import Image
from sklearn.cluster import DBSCAN

# Paths
dataroot = '/mnt/ssd1/divake/nuScenes_tracking_project/datasets'
prediction_file = '/mnt/ssd1/divake/nuScenes_tracking_project/repos/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/prediction.pkl'
output_dir = '/mnt/ssd1/divake/nuScenes_tracking_project/visualizations/detection_videos'
os.makedirs(output_dir, exist_ok=True)

# Class names mapping
CLASS_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer",
               "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]

# Load nuScenes
print("Loading nuScenes dataset...")
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

# Load predictions
print(f"Loading predictions from {prediction_file}...")
with open(prediction_file, 'rb') as f:
    predictions = pickle.load(f)

print(f"Loaded predictions for {len(predictions)} samples")

def get_2d_box_corners(center, size, yaw):
    """Get 2D bounding box corners in bird's eye view

    Args:
        center: [x, y, z] position
        size: [w, l, h] dimensions (width, length, height)
        yaw: rotation angle in radians
    """
    x, y, z = center
    w, l, h = size

    # Create corners (bird's eye view)
    # CenterPoint uses [l/2, w/2] convention
    corners = np.array([
        [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
    ])

    # Rotation matrix
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # Rotate and translate
    corners = corners @ rot_matrix.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners

# Color mapping for different object classes
CLASS_COLORS = {
    'car': '#FF0000',
    'truck': '#FF4500',
    'bus': '#FF8C00',
    'trailer': '#FFA500',
    'construction_vehicle': '#FFD700',
    'pedestrian': '#00CED1',
    'motorcycle': '#1E90FF',
    'bicycle': '#4169E1',
    'traffic_cone': '#32CD32',
    'barrier': '#228B22'
}

def render_box_on_image(ax, box_corners_2d, color, linewidth=2):
    """Render 3D box on camera image"""
    def draw_rect(corners_idx, color):
        prev = corners_idx[-1]
        for corner_idx in corners_idx:
            ax.plot([box_corners_2d[0, prev], box_corners_2d[0, corner_idx]],
                   [box_corners_2d[1, prev], box_corners_2d[1, corner_idx]],
                   color=color, linewidth=linewidth, alpha=0.8)
            prev = corner_idx

    # Draw the 4 vertical lines
    for i in range(4):
        ax.plot([box_corners_2d[0, i], box_corners_2d[0, i + 4]],
               [box_corners_2d[1, i], box_corners_2d[1, i + 4]],
               color=color, linewidth=linewidth, alpha=0.8)

    # Draw bottom and top rectangles
    draw_rect([0, 1, 2, 3], color)  # Bottom
    draw_rect([4, 5, 6, 7], color)  # Top

def detect_objects_from_radar(radar_points, eps=3.0, min_samples=3):
    """
    Simple radar-based object detection using DBSCAN clustering

    Args:
        radar_points: Radar points array (N, 5) - [x, y, z, vx, vy]
        eps: Maximum distance between points in a cluster
        min_samples: Minimum number of points to form a cluster

    Returns:
        List of radar detections with bounding boxes
    """
    if len(radar_points) == 0:
        return []

    # Use only x, y for clustering (bird's eye view)
    points_xy = radar_points[:, :2]

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
    labels = clustering.labels_

    # Generate bounding boxes from clusters
    radar_detections = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:  # Noise points
            continue

        # Get points in this cluster
        cluster_mask = (labels == label)
        cluster_points = radar_points[cluster_mask]

        if len(cluster_points) < min_samples:
            continue

        # Compute bounding box from cluster
        x_min, y_min = cluster_points[:, :2].min(axis=0)
        x_max, y_max = cluster_points[:, :2].max(axis=0)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = cluster_points[:, 2].mean()  # Average z

        width = x_max - x_min + 1.0  # Add padding
        length = y_max - y_min + 1.0
        height = 1.5  # Default height for radar detections

        # Compute average velocity
        vx = cluster_points[:, 3].mean()
        vy = cluster_points[:, 4].mean()

        # Simple orientation from velocity
        yaw = np.arctan2(vy, vx) if (vx**2 + vy**2) > 0.1 else 0.0

        radar_detections.append({
            'center': np.array([center_x, center_y, center_z]),
            'size': np.array([width, length, height]),
            'yaw': yaw,
            'velocity': np.array([vx, vy]),
            'num_points': len(cluster_points),
            'class_name': 'object'  # Generic class for radar
        })

    return radar_detections

def visualize_scene(scene_idx=0, num_frames=40, output_format='mp4'):
    """
    Visualize detections for a scene and create video

    Args:
        scene_idx: Scene index (0-9 for mini dataset)
        num_frames: Number of frames to visualize
        output_format: 'mp4' or 'gif'
    """
    print(f"\nProcessing scene {scene_idx}...")

    # Get scene
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']

    print(f"Scene name: {scene['name']}")
    print(f"Description: {scene['description']}")
    print(f"Total samples: {scene['nbr_samples']}")

    # Create log file
    log_file = os.path.join(output_dir, f'scene_{scene_idx}_log.txt')
    log = open(log_file, 'w')
    log.write(f"{'='*70}\n")
    log.write(f"Detection Visualization Log - Scene {scene_idx}\n")
    log.write(f"{'='*70}\n")
    log.write(f"Scene name: {scene['name']}\n")
    log.write(f"Description: {scene['description']}\n")
    log.write(f"Total samples: {scene['nbr_samples']}\n")
    log.write(f"{'='*70}\n\n")

    # Collect frames
    frames_data = []
    total_detections = 0
    detection_counts_by_class = {}

    for i in range(min(num_frames, scene['nbr_samples'])):
        sample = nusc.get('sample', sample_token)

        # Get LiDAR data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])

        # Load point cloud
        pcl = LidarPointCloud.from_file(lidar_filepath)
        points = pcl.points[:3, :].T  # shape: (N, 3)

        # Get Camera data
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        cam_filepath = os.path.join(dataroot, cam_data['filename'])
        cam_image = Image.open(cam_filepath)

        # Get camera calibration
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

        # Get LiDAR calibration
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        # Get Radar data from all 5 radars
        radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                         'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        all_radar_points = []
        total_radar_points = 0
        for radar_channel in radar_channels:
            radar_token = sample['data'][radar_channel]
            radar_data = nusc.get('sample_data', radar_token)
            radar_filepath = os.path.join(dataroot, radar_data['filename'])
            radar_pcl = RadarPointCloud.from_file(radar_filepath)
            # Get x, y, z, vx, vy (indices 0, 1, 2, 6, 7)
            radar_pts = radar_pcl.points[[0, 1, 2, 6, 7], :].T  # shape: (N, 5) - x, y, z, vx, vy
            all_radar_points.append(radar_pts)
            total_radar_points += radar_pts.shape[0]

        # Concatenate all radar points
        if all_radar_points:
            radar_points = np.vstack(all_radar_points)
        else:
            radar_points = np.zeros((0, 5))

        # Detect objects from radar using DBSCAN clustering
        radar_detections = detect_objects_from_radar(radar_points, eps=3.0, min_samples=3)

        # Get predictions for this sample
        sample_preds = predictions.get(sample['token'], None)

        # Parse predictions from tensor format
        detections = []
        if sample_preds is not None and 'box3d_lidar' in sample_preds:
            boxes = sample_preds['box3d_lidar'].cpu().numpy()  # (N, 9): [x, y, z, w, l, h, vx, vy, yaw]
            scores = sample_preds['scores'].cpu().numpy()  # (N,)
            labels = sample_preds['label_preds'].cpu().numpy().astype(int)  # (N,)

            for j in range(len(boxes)):
                box = boxes[j]
                score = scores[j]
                label_id = labels[j]
                class_name = CLASS_NAMES[label_id]

                center_lidar = box[:3]
                size = box[3:6]
                yaw = box[8]

                # Project to camera for visualization
                rotation = Quaternion(axis=[0, 0, 1], angle=yaw)
                box_lidar = Box(center_lidar, size, rotation)

                # Transform to camera frame
                box_lidar.rotate(Quaternion(lidar_calib['rotation']))
                box_lidar.translate(np.array(lidar_calib['translation']))
                box_lidar.translate(-np.array(cam_calib['translation']))
                box_lidar.rotate(Quaternion(cam_calib['rotation']).inverse)

                # Get corners and project to camera
                corners_3d = box_lidar.corners()
                corners_2d = None

                # Check if box is visible in camera
                if np.all(corners_3d[2, :] > 0):  # In front of camera
                    corners_2d_temp = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
                    if not (np.any(corners_2d_temp[0, :] < 0) or np.any(corners_2d_temp[0, :] >= cam_image.width) or \
                           np.any(corners_2d_temp[1, :] < 0) or np.any(corners_2d_temp[1, :] >= cam_image.height)):
                        corners_2d = corners_2d_temp

                detections.append({
                    'center': box[:3],  # [x, y, z]
                    'size': box[3:6],   # [w, l, h]
                    'yaw': box[8],      # rotation angle
                    'velocity': box[6:8],  # [vx, vy]
                    'class_name': class_name,
                    'score': score,
                    'corners_2d': corners_2d  # For camera rendering
                })

        # Log frame info
        log.write(f"Frame {i+1}/{scene['nbr_samples']}\n")
        log.write(f"  Sample token: {sample['token']}\n")
        log.write(f"  LiDAR points: {len(points):,}\n")
        log.write(f"  Radar points: {total_radar_points} (from 5 radars)\n")
        log.write(f"  LiDAR Detections: {len(detections)}\n")
        log.write(f"  Radar Detections: {len(radar_detections)} (from clustering)\n")

        if len(detections) > 0:
            total_detections += len(detections)
            log.write(f"  Detected objects:\n")
            for j, det in enumerate(detections):
                class_name = det['class_name']
                score = det['score']
                center = det['center']
                log.write(f"    {j+1}. {class_name} (score: {score:.3f}) at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})\n")

                # Count by class
                detection_counts_by_class[class_name] = detection_counts_by_class.get(class_name, 0) + 1
        else:
            log.write(f"  ⚠️  NO DETECTIONS for this frame\n")

        log.write(f"\n")

        frames_data.append({
            'points': points,
            'detections': detections,
            'cam_image': cam_image,
            'radar_points': radar_points,
            'radar_detections': radar_detections,
            'sample_token': sample['token'],
            'frame_idx': i
        })

        # Move to next sample
        if sample['next'] == '':
            break
        sample_token = sample['next']

    print(f"Collected {len(frames_data)} frames")
    print(f"Total detections: {total_detections}")

    # Write summary to log
    log.write(f"\n{'='*70}\n")
    log.write(f"SUMMARY\n")
    log.write(f"{'='*70}\n")
    log.write(f"Total frames: {len(frames_data)}\n")
    log.write(f"Total detections: {total_detections}\n")
    log.write(f"Average detections per frame: {total_detections/len(frames_data) if frames_data else 0:.2f}\n")
    log.write(f"\nDetections by class:\n")
    for cls, count in sorted(detection_counts_by_class.items(), key=lambda x: x[1], reverse=True):
        log.write(f"  {cls}: {count}\n")
    log.write(f"{'='*70}\n")
    log.close()

    print(f"Log saved to: {log_file}")

    # Create figure with three subplots (LiDAR, Camera, Radar)
    fig = plt.figure(figsize=(30, 8), dpi=120)
    ax_lidar = plt.subplot(1, 3, 1)
    ax_camera = plt.subplot(1, 3, 2)
    ax_radar = plt.subplot(1, 3, 3)

    def update_frame(frame_idx):
        ax_lidar.clear()
        ax_camera.clear()
        ax_radar.clear()

        frame = frames_data[frame_idx]
        points = frame['points']
        detections = frame['detections']
        cam_image = frame['cam_image']
        radar_points = frame['radar_points']
        radar_detections = frame['radar_detections']

        # LEFT: LiDAR bird's eye view
        ax_lidar.scatter(points[:, 0], points[:, 1], c=points[:, 2],
                         cmap='viridis', s=0.5, alpha=0.3)

        for det in detections:
            center = det['center']
            size = det['size']
            yaw = det['yaw']
            class_name = det['class_name']
            score = det['score']

            corners = get_2d_box_corners(center, size, yaw)
            color = CLASS_COLORS.get(class_name, '#FFFFFF')
            poly = Polygon(corners, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
            ax_lidar.add_patch(poly)

            ax_lidar.text(center[0], center[1], f'{class_name}\n{score:.2f}',
                         fontsize=6, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

        ax_lidar.set_xlim([-50, 50])
        ax_lidar.set_ylim([-50, 50])
        ax_lidar.set_xlabel('X (meters)', fontsize=10)
        ax_lidar.set_ylabel('Y (meters)', fontsize=10)
        ax_lidar.set_title(f'LiDAR Bird\'s Eye View\nDetections: {len(detections)}',
                          fontsize=12, fontweight='bold')
        ax_lidar.grid(True, alpha=0.3)
        ax_lidar.set_aspect('equal')

        # RIGHT: Camera view with projected boxes
        ax_camera.imshow(cam_image)

        boxes_in_view = 0
        for det in detections:
            if det.get('corners_2d') is not None:
                color = CLASS_COLORS.get(det['class_name'], '#FFFFFF')
                render_box_on_image(ax_camera, det['corners_2d'], color)
                boxes_in_view += 1

        ax_camera.set_xlim([0, cam_image.width])
        ax_camera.set_ylim([cam_image.height, 0])
        ax_camera.set_title(f'Front Camera View\nBoxes in view: {boxes_in_view}',
                           fontsize=12, fontweight='bold')
        ax_camera.axis('off')

        # RIGHT: Radar bird's eye view
        if len(radar_points) > 0:
            # Plot radar points (swap X,Y for bird's eye view: Y=horizontal, X=vertical/forward)
            ax_radar.scatter(radar_points[:, 1], radar_points[:, 0],
                           c='cyan', s=20, alpha=0.8, marker='o', edgecolors='blue', linewidths=0.5)

            # Plot velocity vectors (swap coordinates)
            velocity_scale = 2.0  # Scale factor for visibility
            for radar_pt in radar_points:
                x, y, z, vx, vy = radar_pt
                ax_radar.arrow(y, x, vy*velocity_scale, vx*velocity_scale,
                             head_width=1.5, head_length=1.0, fc='yellow', ec='orange',
                             alpha=0.6, linewidth=1)

        # Show RADAR detection boxes (from clustering)
        for det in radar_detections:
            center = det['center']
            size = det['size']
            yaw = det['yaw']

            corners = get_2d_box_corners(center, size, yaw)
            # Swap X and Y coordinates for corners
            corners_swapped = corners[:, [1, 0]]  # Swap columns
            color = '#00FF00'  # Green for radar detections
            poly = Polygon(corners_swapped, fill=False, edgecolor=color, linewidth=2, alpha=0.9)
            ax_radar.add_patch(poly)

            # Add label showing number of radar points in cluster (swap x,y)
            ax_radar.text(center[1], center[0], f"{det['num_points']}pts",
                         fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))

        ax_radar.set_xlim([-50, 50])
        ax_radar.set_ylim([-50, 50])
        ax_radar.set_xlabel('Y - Left/Right (meters)', fontsize=10)
        ax_radar.set_ylabel('X - Forward (meters)', fontsize=10)
        ax_radar.set_title(f'Radar Detections (DBSCAN)\nClusters: {len(radar_detections)} from {len(radar_points)} points',
                          fontsize=12, fontweight='bold')
        ax_radar.grid(True, alpha=0.3)
        ax_radar.set_aspect('equal')

        fig.suptitle(f'Scene {scene_idx} - Frame {frame_idx+1}/{len(frames_data)}',
                    fontsize=14, fontweight='bold')

    print(f"Creating animation with {len(frames_data)} frames...")
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_data),
                                   interval=200, blit=False, repeat=True)

    # Save video
    if output_format == 'mp4':
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, bitrate=2000, codec='h264')
            output_file = os.path.join(output_dir, f'detections_scene_{scene_idx}_lidar_camera_radar.mp4')
            anim.save(output_file, writer=writer, dpi=150)
            print(f"✓ Video saved to: {output_file}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("Falling back to GIF...")
            output_format = 'gif'

    if output_format == 'gif':
        output_file = os.path.join(output_dir, f'detections_scene_{scene_idx}_lidar_camera_radar.gif')
        anim.save(output_file, writer='pillow', fps=5, dpi=150)
        print(f"✓ GIF saved to: {output_file}")

    plt.close(fig)
    return output_file

if __name__ == '__main__':
    print("=" * 70)
    print("  CenterPoint Detection Visualization")
    print("=" * 70)

    # Note: Scene 0 is training set, predictions only exist for validation scenes
    # Validation scenes in mini: Scene 1 (scene-0103), Scene 6 (scene-0916)

    # Visualize validation scene 1
    print("\n⚠️  NOTE: Predictions only available for validation scenes (1 and 6)")
    print("   Scene 0 is in training set - no predictions available!\n")

    output_file = visualize_scene(scene_idx=1, num_frames=40, output_format='mp4')

    print("\n" + "=" * 70)
    print(f"  ✓ VISUALIZATION COMPLETE!")
    print(f"  Output: {output_file}")
    print("=" * 70)
