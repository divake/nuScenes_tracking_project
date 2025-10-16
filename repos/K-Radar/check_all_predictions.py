'''
Check all predictions including low-confidence ones
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SAMPLE_INDICES = [10]
CONFIDENCE_THR = 0.0  # Check ALL predictions

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
    PATH_MODEL = './pretrained/RTNH_wide_11.pt'

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    import torch
    from torch.utils.data import Subset

    dataset_loaded = pline.dataset_test
    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = pline.dataset_test.collate_fn,
            num_workers = pline.cfg.OPTIMIZER.NUM_WORKERS)

    for dict_item in data_loader:
        dict_item = pline.network(dict_item)

        dataset = pline.dataset_test

        # Get class names
        class_names = []
        dict_label = dataset.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            if logit_idx > 0:
                class_names.append(k)

        # Get ground truth
        labels = dict_item['label'][0]
        print(f"\n=== Frame {SAMPLE_INDICES[0]:03d} ===")
        print(f"\nGround Truth Labels: {len(labels)} vehicles")
        for i, obj in enumerate(labels):
            cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
            print(f"  GT {i+1}: {cls_name:15s} at ({x:6.2f}, {y:6.2f}, {z:6.2f})")

        # Get predictions
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()

        print(f"\nModel Predictions: {len(pred_scores)} total")

        # Sort by confidence score
        sorted_indices = pred_scores.argsort()[::-1]

        for idx in sorted_indices[:20]:  # Show top 20
            score = pred_scores[idx]
            cls_idx = pred_labels[idx]
            cls_name = class_names[cls_idx-1] if cls_idx <= len(class_names) else "Unknown"
            x, y, z, l, w, h, th = pred_boxes[idx]

            status = "✓ SHOWN" if score > 0.5 else "✗ filtered out"
            print(f"  Pred: {cls_name:15s} at ({x:6.2f}, {y:6.2f}, {z:6.2f}) - conf: {score:.3f} {status}")
