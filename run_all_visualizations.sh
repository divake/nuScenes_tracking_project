#!/bin/bash
# Comprehensive visualization runner for nuScenes tracking project

echo "================================================================================"
echo "  nuScenes LiDAR and RADAR Tracking - Complete Visualization Suite"
echo "================================================================================"
echo ""
echo "This script will generate all visualizations for the project:"
echo "  1. LiDAR tracking (10 frames)"
echo "  2. RADAR tracking (10 frames)"
echo "  3. LiDAR vs RADAR comparison (5 frames)"
echo ""
echo "Total output: 25 high-quality visualization frames"
echo ""
echo "================================================================================  "
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py311

# Navigate to project directory
cd /mnt/ssd1/divake/nuScenes_tracking_project

# Verify dataset
echo "Step 1: Verifying nuScenes dataset..."
echo "---------------------------------------"
python test_dataset.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset verification failed!"
    exit 1
fi
echo ""
echo "✓ Dataset verification passed!"
echo ""

# Generate LiDAR visualizations
echo "Step 2: Generating LiDAR tracking visualizations..."
echo "----------------------------------------------------"
python visualize_lidar_tracking.py
if [ $? -ne 0 ]; then
    echo "ERROR: LiDAR visualization failed!"
    exit 1
fi
echo ""
echo "✓ LiDAR visualizations complete!"
echo ""

# Generate RADAR visualizations
echo "Step 3: Generating RADAR tracking visualizations..."
echo "----------------------------------------------------"
python visualize_radar_tracking.py
if [ $? -ne 0 ]; then
    echo "ERROR: RADAR visualization failed!"
    exit 1
fi
echo ""
echo "✓ RADAR visualizations complete!"
echo ""

# Generate comparison visualizations
echo "Step 4: Generating comparison visualizations..."
echo "------------------------------------------------"
python visualize_comparison.py
if [ $? -ne 0 ]; then
    echo "ERROR: Comparison visualization failed!"
    exit 1
fi
echo ""
echo "✓ Comparison visualizations complete!"
echo ""

# Summary
echo "================================================================================"
echo "  ✓✓✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY! ✓✓✓"
echo "================================================================================"
echo ""
echo "Output locations:"
echo "  • LiDAR:      /mnt/ssd1/divake/nuScenes_tracking_project/visualizations/lidar/"
echo "  • RADAR:      /mnt/ssd1/divake/nuScenes_tracking_project/visualizations/radar/"
echo "  • Comparison: /mnt/ssd1/divake/nuScenes_tracking_project/visualizations/comparison/"
echo ""
echo "Total frames generated: 25"
echo ""
echo "To view the visualizations:"
echo "  cd /mnt/ssd1/divake/nuScenes_tracking_project/visualizations"
echo "  eog lidar/lidar_tracking_frame_000.png      # View LiDAR frames"
echo "  eog radar/radar_tracking_frame_000.png      # View RADAR frames"
echo "  eog comparison/comparison_frame_000.png     # View comparison"
echo ""
echo "================================================================================"
echo "  Project: nuScenes LiDAR and RADAR Tracking"
echo "  Status:  ✓ COMPLETE"
echo "  Date:    $(date)"
echo "================================================================================"
