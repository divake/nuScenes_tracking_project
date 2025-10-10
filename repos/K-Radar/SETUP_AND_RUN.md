# K-Radar Setup & Run - Rain Demo
**Working Directory**: `/mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/`
**Goal**: Reproduce their results with provided model and data
**Approach**: Plug and play - use what they provide

---

## 📁 DIRECTORY STRUCTURE (Inside K-Radar/)

```
K-Radar/
├── pretrained/              ← Download models here
│   └── RTNH_wide_11.pt     (will download)
├── data/                    ← Download dataset here
│   └── kradar_dataset/
│       └── XX/             (rain sequence)
├── configs/
│   └── cfg_RTNH_wide.yml   (already exists ✓)
├── main_test_0.py           (visualization script ✓)
├── main_vis.py              (GUI visualization ✓)
└── (other files...)
```

**Everything stays in K-Radar directory** - clean and organized!

---

## 🎯 STEP 1: Download Pretrained Model

**From docs/detection.md, they provide**:
- Model: RTNH_wide (performance: 48.2 AP3D total, 40.3 AP3D in rain!)
- Link: https://drive.google.com/file/d/1ZMtq9BiWCHKKOc20pClCHyhOEI2LLiNM/view?usp=drive_link

### Actions:
```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Create pretrained directory
mkdir -p pretrained

# Download model from Google Drive link above
# Save as: pretrained/RTNH_wide_11.pt
# (File size: ~100-500 MB)

# Verify
ls -lh pretrained/
```

**Expected**: `pretrained/RTNH_wide_11.pt` file exists

---

## 🎯 STEP 2: Download Dataset (Rain Sequence)

**Google Drive Dataset Link**: https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_

### What you need:
1. Find ONE rain sequence (browse folders, check description.txt)
2. Download complete sequence folder

### Directory structure to create:
```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Create data directory
mkdir -p data/kradar_dataset

# After downloading sequence XX from Google Drive
# Place it as:
# data/kradar_dataset/XX/
#   ├── cam_front/
#   ├── cam_left/
#   ├── cam_rear/
#   ├── cam_right/
#   ├── os2-64/           (LiDAR - main)
#   ├── radar_zyx_cube/   (Radar tensor)
#   ├── info_calib/
#   ├── info_label/
#   └── description.txt   (contains "Rain")
```

### Verify:
```bash
# Check structure
ls -la data/kradar_dataset/

# Should see sequence number folder (e.g., 28/)
ls -la data/kradar_dataset/28/

# Verify it's rain
cat data/kradar_dataset/28/description.txt

# Count frames
ls data/kradar_dataset/28/cam_front/ | wc -l
ls data/kradar_dataset/28/os2-64/ | wc -l
ls data/kradar_dataset/28/radar_zyx_cube/ | wc -l
# All should match!
```

---

## 🎯 STEP 3: Install Dependencies

**Use existing env_cu121 (Python 3.8.19 ✓)**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_cu121
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Install requirements
pip install -r requirements.txt

# Install specific versions if needed
pip install open3d==0.15.1
pip install opencv-python==4.2.0.32

# Build Rotated IoU (for detection)
cd utils/Rotated_IoU/cuda_op
python setup.py install
cd ../../..

# Build OpenPCDet operations
cd ops
python setup.py develop
cd ..
```

### Verify installation:
```bash
python -c "import torch; import open3d; import cv2; print('All imports OK!')"
```

---

## 🎯 STEP 4: Prepare Label Files

From docs, they mention:
> Unzip 'kradar_revised_label_v2_0.zip' in the 'tools/revise_label' directory

```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Check if label file exists
ls tools/revise_label/

# If kradar_revised_label_v2_0.zip exists, unzip it
cd tools/revise_label/
unzip kradar_revised_label_v2_0.zip
cd ../..

# If not, might need to download from Google Drive
# Check: https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_
```

---

## 🎯 STEP 5: Configure Paths

**Edit `configs/cfg_RTNH_wide.yml`**:

```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/
nano configs/cfg_RTNH_wide.yml
```

**Find and modify**:
```yaml
# Dataset directory
DIR:
  BASE: '/path/to/kradar_dataset'
  # Change to:
  BASE: '/mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/data/kradar_dataset'

# Test sequences
SEQ:
  TEST: [1, 2, 3, ...]  # Original test sequences
  # Change to your rain sequence:
  TEST: [28]  # Replace 28 with your actual rain sequence number
```

**Save and exit**: Ctrl+O, Enter, Ctrl+X

---

## 🎯 STEP 6: Update Visualization Script

**Edit `main_test_0.py`**:

```bash
nano main_test_0.py
```

**Check/modify lines 16-17**:
```python
PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
PATH_MODEL = './pretrained/RTNH_wide_11.pt'  # Update to match downloaded model name
```

**Check line 10** (sample indices):
```python
SAMPLE_INDICES = [10,11,12,30,70,95,150]
# If your sequence has fewer frames, change to:
SAMPLE_INDICES = [0, 10, 20, 30, 40, 50]
```

**Save and exit**

---

## 🎯 STEP 7: RUN VISUALIZATION!

```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_cu121

# Run visualization
python main_test_0.py
```

### Expected Output:
```
Loading config from ./configs/cfg_RTNH_wide.yml
Loading model from ./pretrained/RTNH_wide_11.pt
Model loaded successfully!
Loading dataset...
Test dataset: 1 sequences
Processing sample 0...
[Open3D window opens]
```

### What you'll see:
- **Open3D 3D viewer** window
- **LiDAR point cloud** (gray/white dots)
- **3D bounding boxes** (colored)
- **Rain scene** with radar detection!
- Multiple frames to navigate

**SUCCESS = Rain demo working!** 🎉

---

## 🎯 OPTIONAL: GUI Visualization

**Try GUI version**:
```bash
python main_vis.py
```

This provides a graphical interface for:
- Selecting sequences
- Viewing multiple sensor data
- Interactive visualization

---

## 🔧 TROUBLESHOOTING

### Error: "Cannot find dataset"
```bash
# Check config file
cat configs/cfg_RTNH_wide.yml | grep -A 2 "DIR:"

# Verify dataset exists
ls data/kradar_dataset/

# Make sure sequence number matches config
ls data/kradar_dataset/28/  # Your sequence
```

### Error: "Cannot load model"
```bash
# Check model file exists
ls -lh pretrained/

# Verify filename matches in main_test_0.py
cat main_test_0.py | grep PATH_MODEL
```

### Error: "Index out of range" (frames)
```bash
# Count available frames
ls data/kradar_dataset/28/cam_front/ | wc -l

# Edit main_test_0.py, line 10
# Use indices within range (0 to num_frames-1)
SAMPLE_INDICES = [0, 10, 20]  # Safe for any sequence
```

### Error: "spconv not found"
```bash
# Install specific version
pip install spconv-cu113==2.1.21

# Or try
conda install -c dglteam spconv
```

### Error: "Rotated_IoU build failed"
```bash
# This might not be needed for just visualization
# Try running without it first

# If needed, check CUDA compatibility
nvcc --version
# Make sure CUDA 11.3 or 11.x available
```

---

## 📋 PRE-FLIGHT CHECKLIST

Before running `python main_test_0.py`:

- [ ] In K-Radar directory: `pwd` → shows `.../repos/K-Radar`
- [ ] Model exists: `ls pretrained/RTNH_wide_11.pt`
- [ ] Dataset exists: `ls data/kradar_dataset/28/` (or your seq #)
- [ ] Config updated: `cat configs/cfg_RTNH_wide.yml | grep BASE`
- [ ] Environment active: `conda env list | grep "*"` → env_cu121
- [ ] Dependencies installed: `python -c "import open3d, torch, cv2"`

**All checked?** → `python main_test_0.py`

---

## 📊 WHAT YOU'LL DEMONSTRATE

**After successful run, you have**:
1. ✅ K-Radar dataset working
2. ✅ Pretrained model loaded
3. ✅ Rain sequence visualized
4. ✅ Radar 3D detection in adverse weather
5. ✅ Demo ready for presentation!

**Key points for demo**:
- Show rain conditions (sparser point cloud)
- Radar still detects objects reliably
- 3D bounding boxes accurate
- Demonstrates robustness vs LiDAR

---

## 🎬 NEXT STEPS (After Success)

1. **Take screenshots** of Open3D visualization
2. **Save frames** (if possible)
3. **Try GUI**: `python main_vis.py`
4. **Optional**: Download fog or snow sequence for comparison

**You'll have a working demo showing**:
- Real-world K-Radar dataset ✓
- State-of-the-art pretrained model ✓
- Rain weather condition ✓
- 4D radar robustness ✓

---

## 📝 SUMMARY

**What we're doing**:
1. Download pretrained RTNH_wide model (~500 MB)
2. Download ONE rain sequence (~2-3 GB)
3. Install dependencies in env_cu121
4. Edit config to point to data
5. Run `main_test_0.py`

**Total time**: 2-4 hours (mostly download time)

**Result**: Working rain demo with radar detection!

**Everything in**: `/mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/`

Clean, organized, reproducible! ✨
