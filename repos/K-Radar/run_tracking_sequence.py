'''
Run detection + tracking on a sequence and create GIF visualization
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import imageio

# Configuration
START_FRAME = 0
NUM_FRAMES = 50  # Number of frames to process
CONFIDENCE_THR = 0.3  # Lower threshold to get more detections
SAVE_DIR = './output_tracking'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
from simple_tracker import SimpleTracker

def get_3d_box_corners(x, y, z, l, w, h, th):
    """Get 3D bounding box corners"""
    corners_x = np.array([l, l, l, l, -l, -l, -l, -l]) / 2
    corners_y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    corners_z = np.array([h, -h, h, -h, h, -h, h, -h]) / 2
    corners = np.row_stack((corners_x, corners_y, corners_z))

    rotation_matrix = np.array([
        [np.cos(th), -np.sin(th), 0.0],
        [np.sin(th), np.cos(th), 0.0],
        [0.0, 0.0, 1.0]])

    corners = rotation_matrix.dot(corners).T + np.array([[x, y, z]])
    return corners

def render_bev_with_tracking(pc, tracks, gt_labels=None, roi=[0,-16,-2,72,16,7.6], resolution=0.2):
    """Render bird's eye view with tracking IDs"""
    x_min, y_min, z_min, x_max, y_max, z_max = roi

    # Create BEV image
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)
    bev_img = np.zeros((y_bins, x_bins, 3), dtype=np.uint8)

    # Plot point cloud
    pc_in_roi = pc[
        (pc[:,0] >= x_min) & (pc[:,0] < x_max) &
        (pc[:,1] >= y_min) & (pc[:,1] < y_max)
    ]

    if len(pc_in_roi) > 0:
        x_img = ((pc_in_roi[:,0] - x_min) / resolution).astype(int)
        y_img = ((pc_in_roi[:,1] - y_min) / resolution).astype(int)
        y_img = y_bins - 1 - y_img  # Flip y-axis
        x_img = np.clip(x_img, 0, x_bins-1)
        y_img = np.clip(y_img, 0, y_bins-1)
        bev_img[y_img, x_img] = [200, 200, 200]

    # Draw ground truth labels (thin gray)
    if gt_labels is not None:
        for obj in gt_labels:
            cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
            corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]

            x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
            y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
            y_pts = y_bins - 1 - y_pts

            pts = np.stack([x_pts, y_pts], axis=1)
            cv2.polylines(bev_img, [pts], True, (80,80,80), 1)

    # Generate unique colors for tracks
    np.random.seed(42)
    colors = {}
    for track_id, bbox, cls_idx, score in tracks:
        if track_id not in colors:
            colors[track_id] = tuple(np.random.randint(100, 255, 3).tolist())

    # Draw tracked predictions with IDs
    for track_id, bbox, cls_idx, score in tracks:
        x, y, z, l, w, h, th = bbox
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]

        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts

        color = colors[track_id]
        pts = np.stack([x_pts, y_pts], axis=1)

        # Draw box with thicker line
        cv2.polylines(bev_img, [pts], True, color, 3)

        # Draw arrow showing direction
        center_x = int((x - x_min) / resolution)
        center_y = y_bins - 1 - int((y - y_min) / resolution)
        arrow_len = int(l / resolution)
        end_x = int(center_x + arrow_len * np.cos(th))
        end_y = int(center_y - arrow_len * np.sin(th))

        if 0 <= center_x < x_bins and 0 <= center_y < y_bins:
            cv2.arrowedLine(bev_img, (center_x, center_y), (end_x, end_y),
                          color, 2, tipLength=0.3)

            # Draw track ID
            label = f"ID:{track_id}"
            cv2.putText(bev_img, label, (center_x + 10, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return bev_img

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_11.pt'

    # Create output directory
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    import torch
    from torch.utils.data import Subset

    # Create tracker
    tracker = SimpleTracker(max_age=5, min_hits=1, iou_threshold=0.2)

    dataset_loaded = pline.dataset_test

    # Use sequential frames
    frame_indices = list(range(START_FRAME, START_FRAME + NUM_FRAMES))
    subset = Subset(dataset_loaded, frame_indices)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = pline.dataset_test.collate_fn,
            num_workers = 0)  # Set to 0 for sequential processing

    print(f"Processing {NUM_FRAMES} frames for tracking...")

    frame_files = []
    for idx, dict_item in enumerate(tqdm(data_loader)):
        # Run detection
        dict_item = pline.network(dict_item)

        dataset = pline.dataset_test
        pc_lidar = dataset.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])

        # Get predictions
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        # Filter by confidence
        conf_mask = pred_scores > CONFIDENCE_THR
        filtered_boxes = pred_boxes[conf_mask]
        filtered_scores = pred_scores[conf_mask]
        filtered_labels = pred_labels[conf_mask]

        # Update tracker
        tracks = tracker.update(filtered_boxes, filtered_scores, filtered_labels)

        # Get ground truth for comparison
        gt_labels = dict_item['label'][0]

        # Render frame
        bev_img = render_bev_with_tracking(pc_lidar, tracks, gt_labels)

        # Add frame number
        cv2.putText(bev_img, f"Frame: {START_FRAME + idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bev_img, f"Tracks: {len(tracks)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save frame
        frame_path = f'{SAVE_DIR}/frame_{idx:03d}.png'
        cv2.imwrite(frame_path, bev_img)
        frame_files.append(frame_path)

    # Create GIF
    print("\nCreating GIF...")
    images = []
    for frame_path in frame_files:
        images.append(imageio.imread(frame_path))

    gif_path = f'{SAVE_DIR}/tracking_result.gif'
    imageio.mimsave(gif_path, images, duration=0.1, loop=0)

    print(f"\n✓ Tracking complete!")
    print(f"  Frames saved: {SAVE_DIR}/frame_*.png")
    print(f"  GIF created: {gif_path}")
    print(f"  Total tracks created: {tracker.tracks[-1].id if tracker.tracks else 0}")
