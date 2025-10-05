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
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import torch

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

                detections.append({
                    'center': box[:3],  # [x, y, z]
                    'size': box[3:6],   # [w, l, h]
                    'yaw': box[8],      # rotation angle
                    'velocity': box[6:8],  # [vx, vy]
                    'class_name': class_name,
                    'score': score
                })

        # Log frame info
        log.write(f"Frame {i+1}/{scene['nbr_samples']}\n")
        log.write(f"  Sample token: {sample['token']}\n")
        log.write(f"  LiDAR points: {len(points):,}\n")
        log.write(f"  Detections: {len(detections)}\n")

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

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    def update_frame(frame_idx):
        ax.clear()

        frame = frames_data[frame_idx]
        points = frame['points']
        detections = frame['detections']

        # Plot point cloud (bird's eye view)
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2],
                   cmap='viridis', s=0.5, alpha=0.3)

        # Plot predicted bounding boxes
        for det in detections:
            center = det['center']
            size = det['size']
            yaw = det['yaw']
            class_name = det['class_name']
            score = det['score']

            # Get 2D box corners
            corners = get_2d_box_corners(center, size, yaw)

            # Get color
            color = CLASS_COLORS.get(class_name, '#FFFFFF')

            # Draw box
            poly = Polygon(corners, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
            ax.add_patch(poly)

            # Add label
            ax.text(center[0], center[1], f'{class_name}\n{score:.2f}',
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

        # Set limits and labels
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'CenterPoint Detections - Scene {scene_idx} - Frame {frame_idx+1}/{len(frames_data)}\n'
                    f'Detections: {len(detections)} | Points: {len(points):,}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        return ax.get_children()

    print(f"Creating animation with {len(frames_data)} frames...")
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_data),
                                   interval=200, blit=False, repeat=True)

    # Save video
    if output_format == 'mp4':
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, bitrate=2000, codec='h264')
            output_file = os.path.join(output_dir, f'detections_scene_{scene_idx}.mp4')
            anim.save(output_file, writer=writer, dpi=150)
            print(f"✓ Video saved to: {output_file}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("Falling back to GIF...")
            output_format = 'gif'

    if output_format == 'gif':
        output_file = os.path.join(output_dir, f'detections_scene_{scene_idx}.gif')
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
