'''
Create 3-panel visualization: Camera + LiDAR BEV + Radar BEV
Simplified version for demo
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio

# Configuration
START_FRAME = 1  # Start from frame 1 (cameras start at 00001)
NUM_FRAMES = 50
CONFIDENCE_THR = 0.3
OUTPUT_DIR = './demo_3panel'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

def get_3d_box_corners(x, y, z, l, w, h, th):
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

def render_lidar_bev(pc, labels, preds, roi=[0,-16,-2,72,16,7.6], resolution=0.1):
    x_min, y_min, z_min, x_max, y_max, z_max = roi
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)
    bev_img = np.zeros((y_bins, x_bins, 3), dtype=np.uint8)

    # Plot LiDAR
    pc_in_roi = pc[(pc[:,0] >= x_min) & (pc[:,0] < x_max) & (pc[:,1] >= y_min) & (pc[:,1] < y_max)]
    if len(pc_in_roi) > 0:
        x_img = ((pc_in_roi[:,0] - x_min) / resolution).astype(int)
        y_img = ((pc_in_roi[:,1] - y_min) / resolution).astype(int)
        y_img = y_bins - 1 - y_img
        x_img = np.clip(x_img, 0, x_bins-1)
        y_img = np.clip(y_img, 0, y_bins-1)
        # Gradient coloring by height
        z_norm = np.clip((pc_in_roi[:,2] + 2) / 5 * 150 + 50, 50, 200).astype(np.uint8)
        bev_img[y_img, x_img] = np.stack([z_norm, z_norm, z_norm], axis=-1)

    # Draw GT (orange)
    for obj in labels:
        cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]
        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts
        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, (0,50,255), 2)

    # Draw predictions (green)
    for pred_box, score, cls_idx in preds:
        x, y, z, l, w, h, th = pred_box
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]
        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts
        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, (0,255,0), 3)

    return bev_img

def render_radar_bev(rdr_sparse, labels, preds, roi=[0,-16,-2,72,16,7.6], resolution=0.1):
    x_min, y_min, z_min, x_max, y_max, z_max = roi
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)

    # Blue background
    bev_img = np.zeros((y_bins, x_bins, 3), dtype=np.uint8)
    bev_img[:,:,0] = 30  # Subtle blue

    # Plot radar points
    if len(rdr_sparse) > 0:
        rdr_in_roi = rdr_sparse[(rdr_sparse[:,0] >= x_min) & (rdr_sparse[:,0] < x_max) & (rdr_sparse[:,1] >= y_min) & (rdr_sparse[:,1] < y_max)]
        if len(rdr_in_roi) > 0:
            x_img = ((rdr_in_roi[:,0] - x_min) / resolution).astype(int)
            y_img = ((rdr_in_roi[:,1] - y_min) / resolution).astype(int)
            y_img = y_bins - 1 - y_img
            x_img = np.clip(x_img, 0, x_bins-1)
            y_img = np.clip(y_img, 0, y_bins-1)

            power = rdr_in_roi[:,3] if rdr_in_roi.shape[1] > 3 else np.ones(len(rdr_in_roi))
            power_norm = ((power - power.min()) / (power.max() - power.min() + 1e-6) * 200 + 55).astype(np.uint8)
            bev_img[y_img, x_img, 0] = 255  # Blue
            bev_img[y_img, x_img, 1] = power_norm  # Green gradient by power

    # Draw GT (orange)
    for obj in labels:
        cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]
        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts
        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, (0,50,255), 2)

    # Draw predictions (green)
    for pred_box, score, cls_idx in preds:
        x, y, z, l, w, h, th = pred_box
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]
        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts
        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, (0,255,0), 3)

    return bev_img

def add_2d_boxes_to_camera(img, labels, preds):
    """Add simple 2D boxes to camera (simplified for demo)"""
    img_out = img.copy()

    # Draw GT boxes (orange)
    for obj in labels:
        cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
        if 10 < x < 60 and -8 < y < 8:  # Rough visibility filter
            # Simple projection approximation
            img_x = int(640 + y * 40)
            img_y = int(360 - x * 5)
            box_w = int(l * 20)
            box_h = int(h * 30)
            x1 = max(0, img_x - box_w//2)
            y1 = max(0, img_y - box_h//2)
            x2 = min(1280, img_x + box_w//2)
            y2 = min(720, img_y + box_h//2)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0,50,255), 2)

    # Draw prediction boxes (green)
    for pred_box, score, cls_idx in preds:
        x, y, z, l, w, h, th = pred_box
        if 10 < x < 60 and -8 < y < 8:
            img_x = int(640 + y * 40)
            img_y = int(360 - x * 5)
            box_w = int(l * 20)
            box_h = int(h * 30)
            x1 = max(0, img_x - box_w//2)
            y1 = max(0, img_y - box_h//2)
            x2 = min(1280, img_x + box_w//2)
            y2 = min(720, img_y + box_h//2)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.putText(img_out, f'{score:.2f}', (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return img_out

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_11.pt'

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    import torch
    from torch.utils.data import Subset

    dataset_loaded = pline.dataset_test
    frame_indices = list(range(START_FRAME, START_FRAME + NUM_FRAMES))
    subset = Subset(dataset_loaded, frame_indices)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size=1, shuffle=False,
            collate_fn=pline.dataset_test.collate_fn,
            num_workers=0)

    print(f"Processing {NUM_FRAMES} frames...")
    frame_files = []

    for idx, dict_item in enumerate(tqdm(data_loader)):
        dataset = pline.dataset_test

        # Get radar sparse data BEFORE network processing
        rdr_sparse_tensor = dict_item['rdr_sparse']  # Nx4 tensor (flattened)
        batch_indices = dict_item['batch_indices_rdr_sparse']  # Batch indices
        # Get points for first item in batch
        rdr_sparse_input = rdr_sparse_tensor[batch_indices == 0].detach().cpu().numpy()

        # Run detection
        dict_item = pline.network(dict_item)

        pc_lidar = dataset.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])

        # Load camera - extract frame number from lidar path
        lidar_path = dict_item['meta'][0]['path']['ldr64']
        frame_num = lidar_path.split('_')[-1].split('.')[0]  # Get frame number
        header = dict_item['meta'][0]['header']
        seq = dict_item['meta'][0]['seq']
        cam_path = f"{header}/{seq}/cam-front/cam-front_{frame_num}.png"

        cam_img = cv2.imread(cam_path)
        if cam_img is None:
            print(f"Warning: Could not load {cam_path}")
            continue

        # Camera images are 2560x720, but only first 1280 is the actual front camera
        cam_img = cam_img[:, :1280]  # Take only first half

        # Get predictions
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        conf_mask = pred_scores > CONFIDENCE_THR
        filtered_preds = list(zip(pred_boxes[conf_mask], pred_scores[conf_mask], pred_labels[conf_mask]))

        gt_labels = dict_item['label'][0]

        # Render views
        lidar_bev = render_lidar_bev(pc_lidar, gt_labels, filtered_preds)
        radar_bev = render_radar_bev(rdr_sparse_input, gt_labels, filtered_preds)
        # Don't add boxes to camera - they need proper calibration
        camera_img = cam_img.copy()

        # Resize to same height
        target_h = 400
        h_cam, w_cam, _ = camera_img.shape
        h_ldr, w_ldr, _ = lidar_bev.shape
        h_rdr, w_rdr, _ = radar_bev.shape

        camera_img = cv2.resize(camera_img, (int(w_cam * target_h / h_cam), target_h))
        lidar_bev = cv2.resize(lidar_bev, (int(w_ldr * target_h / h_ldr), target_h))
        radar_bev = cv2.resize(radar_bev, (int(w_rdr * target_h / h_rdr), target_h))

        # Add labels
        cv2.putText(camera_img, "Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(lidar_bev, "LiDAR BEV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(radar_bev, "Radar BEV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Concatenate
        combined = cv2.hconcat([camera_img, lidar_bev, radar_bev])

        # Add frame number
        cv2.putText(combined, f"Frame: {START_FRAME + idx}", (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Save
        frame_path = f'{OUTPUT_DIR}/frame_{idx:03d}.png'
        cv2.imwrite(frame_path, combined)
        frame_files.append(frame_path)

    # Create GIF
    print("\nCreating GIF...")
    images = [imageio.imread(fp) for fp in frame_files]
    gif_path = f'{OUTPUT_DIR}/kradar_demo.gif'
    imageio.mimsave(gif_path, images, duration=100, loop=0)

    print(f"\n✓ Demo complete!")
    print(f"  GIF: {gif_path}")
    print(f"  Format: Camera | LiDAR BEV | Radar BEV")
    print(f"  Orange boxes = Ground Truth")
    print(f"  Green boxes = RTNH_wide Detections")
