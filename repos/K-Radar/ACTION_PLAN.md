# K-Radar Implementation - FINAL ACTION PLAN
**Date**: 2025-10-07
**Goal**: Get ONE sequence working with pretrained model

---

## 🔍 KEY FINDINGS

### 1. **Train/Test Split Analysis** ✅
- Checked `resources/split/train.txt` and `resources/split/test.txt`
- **ALL 58 sequences appear in BOTH files!**
- This means they use frame-level splitting, not sequence-level splitting
- **Any sequence can be used for validation/testing**

### 2. **Google Drive Structure** ✅
You found these folders:
```
Google Drive:
├── 1-20/          ← Sequences 1 to 20
├── 21-37/         ← Sequences 21 to 37
├── 38-58/         ← Sequences 38 to 58
├── Doppler/
├── Etc/
├── IMU/
├── Pretrained/    ← Models here!
├── ProcessedData/
├── RadarTensor/
└── RTNHP_TLP/
```

### 3. **Sequence Selection** ✅
- **Choose: Sequence 30** (middle range, good for testing)
- Located in: `1-20/` folder? No, wait... 30 is > 20
- Actually in: `21-37/` folder
- **Better choice: Sequence 15** (in `1-20/` folder, truly middle)

### 4. **Existing Packages in env_cu121** ✅
Already installed:
- ✅ easydict 1.13
- ✅ opencv-python 4.10.0 (but need 4.2.0.32 - will handle)
- ✅ torch 2.4.1 (newer than required 1.11.0 - will test first)
- ❌ open3d - **NEED TO INSTALL**
- ❌ spconv - **NEED TO INSTALL**

Missing from requirements.txt:
- open3d==0.15.1
- spconv-cu113
- tensorboard
- scikit-image
- numba
- einops
- PyQt5
- setuptools==59.5.0

### 5. **Pretrained Model Location** ✅
From Google Drive: `Pretrained/` folder
Model name: `RTNH_wide_11.pt` (from docs/detection.md)

---

## 📋 STEP-BY-STEP ACTION PLAN

### STEP 1: Install Missing Dependencies (15-30 min)

**Already in env_cu121 ✅**:
- easydict
- opencv (will try with 4.10.0 first, downgrade if needed)
- torch 2.4.1 (try first, downgrade if needed)

**Need to install**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_cu121
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Core visualization (CRITICAL)
pip install open3d==0.15.1

# Sparse convolution (CRITICAL for model)
pip install spconv-cu113

# Supporting packages
pip install tensorboard
pip install scikit-image
pip install numba
pip install einops
pip install PyQt5

# Check setuptools version
pip show setuptools
# If not 59.5.0, consider downgrading:
# pip install setuptools==59.5.0
```

**Test after install**:
```bash
python -c "import open3d; print(f'Open3D: {open3d.__version__}')"
python -c "import spconv; print('spconv OK')"
```

---

### STEP 2: Download Pretrained Model (10-15 min)

**From Google Drive**:
- Folder: `Pretrained/`
- File: `RTNH_wide_11.pt` (or similar - check folder contents)

**Download to**:
```bash
# Create pretrained directory
mkdir -p /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/pretrained/

# After downloading from Google Drive:
# Move to: pretrained/RTNH_wide_11.pt
```

**Verify**:
```bash
ls -lh pretrained/
# Should show: RTNH_wide_11.pt (~100-500 MB)
```

---

### STEP 3: Download ONE Sequence (1-2 hours)

**Recommendation: Sequence 15** (or 20 if 15 not accessible)

**From Google Drive**:
- Folder: `1-20/15/` (or `1-20/20/`)

**What to download**:
Complete sequence folder containing:
- `cam_front/`
- `cam_left/`
- `cam_rear/`
- `cam_right/`
- `os2-64/` (LiDAR - main)
- `radar_zyx_cube/` (Radar tensor - CRITICAL)
- `info_calib/`
- `info_label/`
- `description.txt`

**Download to**:
```bash
mkdir -p /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/data/kradar_dataset/

# After downloading from Google Drive, structure should be:
# data/kradar_dataset/15/
#   ├── cam_front/
#   ├── cam_left/
#   ├── cam_rear/
#   ├── cam_right/
#   ├── os2-64/
#   ├── radar_zyx_cube/
#   ├── info_calib/
#   ├── info_label/
#   └── description.txt
```

**Verify**:
```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Check structure
ls -la data/kradar_dataset/15/

# Read description
cat data/kradar_dataset/15/description.txt

# Count frames (should all match)
ls data/kradar_dataset/15/cam_front/ | wc -l
ls data/kradar_dataset/15/os2-64/ | wc -l
ls data/kradar_dataset/15/radar_zyx_cube/ | wc -l
```

---

### STEP 4: Build Required Packages (15-30 min)

**Rotated IoU** (for detection):
```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/
cd utils/Rotated_IoU/cuda_op
python setup.py install
cd ../../..
```

**OpenPCDet operations**:
```bash
cd ops
python setup.py develop
cd ..
```

**If builds fail**:
- Check CUDA version compatibility
- May need to skip if only visualizing (not training)

---

### STEP 5: Configure Paths (5-10 min)

**Edit `configs/cfg_RTNH_wide.yml`**:

```bash
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/
nano configs/cfg_RTNH_wide.yml
```

**Find line 35-36 and change**:
```yaml
# OLD:
list_dir_kradar: ['/media/donghee/kradar/dataset']

# NEW:
list_dir_kradar: ['/mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/data/kradar_dataset']
```

**Find line 64 and change**:
```yaml
# OLD:
dir: '/media/donghee/kradar/rdr_sparse_data/rtnh_wider_1p_1'

# NEW - THIS IS IMPORTANT:
# We downloaded radar_zyx_cube, not preprocessed sparse data
# Need to check if we need to set 'processed': False
# OR download processed data from ProcessedData folder

# For now, let's note this might need adjustment
```

**Save**: Ctrl+O, Enter, Ctrl+X

---

### STEP 6: Update main_test_0.py (5 min)

```bash
nano main_test_0.py
```

**Check/Update**:
```python
# Line 10-11: Sample indices
SAMPLE_INDICES = [10,11,12,30,70,95,150]
# If sequence has fewer frames, change to:
# SAMPLE_INDICES = [0, 10, 20, 30]

# Line 16-17: Paths
PATH_CONFIG = './configs/cfg_RTNH_wide.yml'
PATH_MODEL = './pretrained/RTNH_wide_11.pt'  # Match downloaded filename
```

---

### STEP 7: RUN IT! (5-10 min)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_cu121
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Run visualization
python main_test_0.py
```

**Expected Output**:
```
Loading config...
Loading model from ./pretrained/RTNH_wide_11.pt
Model loaded!
Loading dataset...
Processing frame 10...
[Open3D window opens]
```

---

## ⚠️ POTENTIAL ISSUES & SOLUTIONS

### Issue 1: Preprocessed Radar Data Not Found
```
Error: Cannot find rdr_sparse data at /media/donghee/kradar/rdr_sparse_data/
```

**Solution**:
We downloaded `radar_zyx_cube` but config expects preprocessed sparse data.

**Options**:
A) Download from `ProcessedData/` folder in Google Drive
B) Modify config to use raw radar_zyx_cube:
   ```yaml
   rdr_sparse:
     processed: False  # Change from True
     dir: ''  # Empty
   rdr:
     cube: True  # Enable raw cube
   ```

### Issue 2: PyTorch Version Mismatch
```
Error: Model trained with PyTorch 1.11, you have 2.4.1
```

**Solution**:
```bash
conda activate env_cu121
pip uninstall torch torchvision torchaudio
pip install torch==1.11.0+cu113 torchvision==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Issue 3: OpenCV Version Issues
```
Error related to cv2 functions
```

**Solution**:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.2.0.32
```

### Issue 4: Spconv Installation Fails
```
Error building spconv-cu113
```

**Solution**:
```bash
# Try specific version
pip install spconv-cu113==2.1.21

# Or conda
conda install -c dglteam spconv
```

### Issue 5: Build Errors (Rotated_IoU or ops)
```
CUDA compilation errors
```

**Solution**:
- Check CUDA version: `nvcc --version`
- These might not be needed for just visualization
- Try running without building first

---

## 📊 WHAT TO DOWNLOAD FROM GOOGLE DRIVE

### Priority 1 - MUST HAVE:
1. **Pretrained Model**
   - Folder: `Pretrained/`
   - File: `RTNH_wide_11.pt`
   - Size: ~100-500 MB

2. **ONE Sequence (recommend 15 or 20)**
   - Folder: `1-20/15/` (or `1-20/20/`)
   - All subfolders (cam_*, os2-64, radar_zyx_cube, info_*)
   - Size: ~2-3 GB

### Priority 2 - MIGHT NEED:
3. **Processed Radar Data** (if raw radar doesn't work)
   - Folder: `ProcessedData/`
   - Look for: `rtnh_wider_1p_1/` or similar
   - This is preprocessed sparse radar data

---

## 🎯 SIMPLIFIED DOWNLOAD CHECKLIST

**From Google Drive** (https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_):

- [ ] Navigate to `Pretrained/` folder
- [ ] Download `RTNH_wide_11.pt` → save to `K-Radar/pretrained/`
- [ ] Navigate to `1-20/` folder
- [ ] Find sequence 15 (or 20)
- [ ] Download entire sequence folder
- [ ] Extract to `K-Radar/data/kradar_dataset/15/`
- [ ] Verify all subfolders present

**Optional** (if radar data issues):
- [ ] Navigate to `ProcessedData/` folder
- [ ] Look for preprocessed radar data
- [ ] Download if needed

---

## 📝 EXECUTION ORDER

**Do in this exact order**:

1. ✅ Install dependencies (STEP 1)
2. ✅ Download pretrained model (STEP 2)
3. ✅ Download sequence 15 (STEP 3)
4. ⚠️ Try running first (STEP 5-7)
5. ⚠️ If fails, build packages (STEP 4)
6. ⚠️ If still fails, debug (see potential issues)

**Why skip builds initially?**
- They might not be needed for just visualization
- Faster to test if basic setup works
- Can build later if needed

---

## 🚀 QUICK START (After Downloads)

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_cu121

# Go to K-Radar
cd /mnt/ssd1/divake/nuScenes_tracking_project/repos/K-Radar/

# Install critical packages
pip install open3d==0.15.1 spconv-cu113

# Edit config
nano configs/cfg_RTNH_wide.yml
# Change line 35: list_dir_kradar to point to ./data/kradar_dataset

# Edit main_test_0.py if needed
nano main_test_0.py
# Check PATH_MODEL matches downloaded model name

# RUN!
python main_test_0.py
```

---

## ✅ SUCCESS CRITERIA

**You'll know it works when**:
- Script loads without errors
- Open3D window appears
- Shows 3D point cloud (LiDAR)
- Shows 3D bounding boxes
- Can see detections from radar
- Can navigate frames

**Then you have**: Working K-Radar demo! 🎉

---

## 📌 NOTES

- **Sequence 15 chosen**: Middle of range, likely good data
- **Alternative: Sequence 20**: Also in `1-20/` folder
- **All sequences work**: No specific train/val/test split at sequence level
- **env_cu121 reuse**: Already has Python 3.8.19, most dependencies
- **Everything in K-Radar/**: Clean, organized, no external dependencies

Ready to execute! Start with STEP 1 (install dependencies).
