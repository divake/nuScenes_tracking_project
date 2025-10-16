import sys
sys.path.append('./')

from easydict import EasyDict
import yaml

# Load config
cfg_path = './configs/sparse_rdr_data_generation/cfg_gen_wider_rtnh_10p.yml'
with open(cfg_path, 'r') as f:
    cfg = EasyDict(yaml.safe_load(f))

# Import dataset after path is set
from datasets.kradar_detection_v1_1 import KRadarDetection_v1_1

# Generate sparse data for test split (which includes sequence 1)
print("Generating sparse radar data for RTNH_wide...")
dataset = KRadarDetection_v1_1(cfg=cfg, split='test')
dataset.generate_sparse_rdr_cube_for_wider_rtnh(vis_for_check=False, vis_in_sampled_sphere=False)

print("\nDone! Sparse radar data generated.")
