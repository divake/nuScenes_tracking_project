# nuScenes LiDAR and RADAR Tracking Project

**Status**: ✅ COMPLETE & WORKING
**Date**: 2025-10-04
**Dataset**: nuScenes v1.0-mini

---

## Quick Start

### 1. Run All Visualizations
```bash
cd ~/nuScenes_tracking_project
./run_all_visualizations.sh
```

This will generate:
- 10 LiDAR tracking visualizations
- 10 RADAR tracking visualizations
- 5 comparison visualizations

### 2. Run Individual Scripts

**LiDAR Tracking:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py311
python visualize_lidar_tracking.py
```

**RADAR Tracking:**
```bash
python visualize_radar_tracking.py
```

**Comparison:**
```bash
python visualize_comparison.py
```

**Dataset Test:**
```bash
python test_dataset.py
```

---

## Project Structure

```
nuScenes_tracking_project/
├── README.md                          ← You are here
├── SETUP_COMPLETE.md                  ← Comprehensive documentation
├── PROGRESS_REPORT.md                 ← Detailed progress tracking
├── run_all_visualizations.sh          ← Run everything
│
├── visualize_lidar_tracking.py        ← LiDAR visualization script
├── visualize_radar_tracking.py        ← RADAR visualization script
├── visualize_comparison.py            ← Comparison script
├── test_dataset.py                    ← Dataset verification
│
├── visualizations/                    ← Generated outputs
│   ├── lidar/                         ← 10 LiDAR frames
│   ├── radar/                         ← 10 RADAR frames
│   └── comparison/                    ← 5 comparison frames
│
├── datasets/                          ← nuScenes data (3.9 GB)
├── repos/                             ← CenterPoint & AB3DMOT repos
└── [other files]
```

---

## What This Project Does

✅ **LiDAR Point Cloud Visualization**
- High-density 3D point clouds (~35,000 points/frame)
- 3D bounding boxes with track IDs
- Bird's eye view and front view
- Color-coded by object category

✅ **RADAR Point Cloud Visualization**
- Sparse point clouds (~250 points/frame)
- Velocity information and vectors
- Range-azimuth plots
- Weather-resistant sensor characteristics

✅ **LiDAR vs RADAR Comparison**
- Side-by-side sensor comparison
- Statistical analysis
- Sensor fusion insights

✅ **Object Tracking**
- Persistent track IDs across frames
- 69-124 objects tracked in LiDAR
- Multiple object categories (vehicles, pedestrians, etc.)

---

## Key Results

### LiDAR
- **Points per frame**: ~34,700
- **Tracked objects**: 69-124 per frame
- **Unique tracks**: 124
- **Range**: ~50m
- **Advantages**: High spatial resolution, accurate 3D geometry

### RADAR
- **Points per frame**: ~250
- **Velocity data**: Direct measurement available
- **Range**: ~80m
- **Advantages**: Weather resistant, long range, velocity information

---

## Documentation

- **SETUP_COMPLETE.md**: Comprehensive setup and results documentation
- **PROGRESS_REPORT.md**: Detailed progress tracking
- **installation_log.md**: Package installation details
- **setup_notes.md**: Repository analysis notes

---

## Requirements

- Python 3.11
- CUDA 12.1
- nuScenes dataset v1.0-mini (3.9 GB)
- See `installed_packages.txt` for complete package list

---

## View Visualizations

```bash
cd ~/nuScenes_tracking_project/visualizations

# View LiDAR frames
ls lidar/

# View RADAR frames
ls radar/

# View comparison frames
ls comparison/
```

To view images (using Eye of GNOME or any image viewer):
```bash
eog lidar/lidar_tracking_frame_000.png
eog radar/radar_tracking_frame_000.png
eog comparison/comparison_frame_000.png
```

---

## Next Steps

1. ✅ **Current**: Basic visualization and tracking working
2. 🔄 **Future**: Download CenterPoint pretrained model
3. 🔄 **Future**: Run full detection pipeline
4. 🔄 **Future**: Implement sensor fusion
5. 🔄 **Future**: Real-time tracking

---

## Credits

- **Dataset**: nuScenes (https://www.nuscenes.org/)
- **CenterPoint**: https://github.com/tianweiy/CenterPoint
- **AB3DMOT**: https://github.com/xinshuoweng/AB3DMOT

---

## License

This project uses the nuScenes dataset which has its own license.
See: https://www.nuscenes.org/terms-of-use

---

**Generated**: 2025-10-04
**Status**: ✅ COMPLETE & WORKING
**Total Visualizations**: 25 frames
