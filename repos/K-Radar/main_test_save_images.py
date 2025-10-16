'''
* Modified version to save detection images instead of interactive visualization
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from pathlib import Path

SAMPLE_INDICES = [10,11,12,30,70,95,150]
CONFIDENCE_THR = 0.5
SAVE_DIR = './output_detections'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

def project_3d_to_2d(corners, calib_proj):
    """Project 3D corners to 2D image coordinates"""
    # corners: Nx3, calib_proj: 3x4
    corners_homo = np.hstack([corners, np.ones((corners.shape[0], 1))])  # Nx4
    pts_2d = corners_homo @ calib_proj.T  # Nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2].astype(int)

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

def draw_3d_box_on_image(img, x, y, z, l, w, h, th, proj_matrix, color):
    """Draw 3D bounding box on image"""
    corners = get_3d_box_corners(x, y, z, l, w, h, th)
    pts_2d = project_3d_to_2d(corners, proj_matrix)

    # Draw box edges
    edges = [[0,1], [0,2], [1,3], [2,3], [4,5], [4,6], [5,7], [6,7],
             [0,4], [1,5], [2,6], [3,7]]

    for edge in edges:
        pt1 = tuple(pts_2d[edge[0]])
        pt2 = tuple(pts_2d[edge[1]])
        # Check if points are within image bounds
        h, w = img.shape[:2]
        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(img, pt1, pt2, color, 2)

    return img

def render_bev(pc, labels, preds, roi=[0,-16,-2,72,16,7.6], resolution=0.2):
    """Render bird's eye view"""
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

    # Draw labels (gray)
    for obj in labels:
        cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]  # Get XY only

        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts  # Flip y-axis

        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, (128,128,128), 2)

    # Draw predictions (green for Sedan, orange for Bus/Truck)
    for pred_box, score, cls_idx in preds:
        x, y, z, l, w, h, th = pred_box
        corners2d = get_3d_box_corners(x, y, z, l, w, h, th)[:,[0,1]]

        x_pts = ((corners2d[[0,2,6,4],0] - x_min) / resolution).astype(int)
        y_pts = ((corners2d[[0,2,6,4],1] - y_min) / resolution).astype(int)
        y_pts = y_bins - 1 - y_pts

        color = (0,255,0) if cls_idx == 1 else (0,50,255)  # Green or Orange
        pts = np.stack([x_pts, y_pts], axis=1)
        cv2.polylines(bev_img, [pts], True, color, 2)

    return bev_img

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_11.pt'

    # Create output directory
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
    pline.load_dict_model(PATH_MODEL)

    pline.network.eval()

    import torch
    from tqdm import tqdm
    from torch.utils.data import Subset

    dataset_loaded = pline.dataset_test
    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = pline.dataset_test.collate_fn,
            num_workers = pline.cfg.OPTIMIZER.NUM_WORKERS)

    print(f"Processing {len(subset)} samples...")

    for idx, dict_item in enumerate(tqdm(data_loader)):
        dict_item = pline.network(dict_item)

        dataset = pline.dataset_test
        pc_lidar = dataset.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])

        class_names = []
        dict_label = dataset.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            if logit_idx > 0:
                class_names.append(k)

        # Get predictions
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        # Filter by confidence
        conf_mask = pred_scores > CONFIDENCE_THR
        filtered_preds = list(zip(pred_boxes[conf_mask], pred_scores[conf_mask], pred_labels[conf_mask]))

        # Get ground truth labels
        labels = dict_item['label'][0]

        # Create BEV visualization
        bev_img = render_bev(pc_lidar, labels, filtered_preds)

        # Save BEV image
        sample_idx = SAMPLE_INDICES[idx]
        cv2.imwrite(f'{SAVE_DIR}/detection_bev_{sample_idx:03d}.png', bev_img)

        print(f"Saved detection_bev_{sample_idx:03d}.png ({len(filtered_preds)} detections)")

    print(f"\nAll detections saved to {SAVE_DIR}/")
