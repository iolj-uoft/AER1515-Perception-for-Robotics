# Assignment 3 — Depth Estimation, Object Detection & Instance Segmentation

This assignment covers three main tasks:
1. **Dense Depth Estimation** from stereo disparity maps
2. **Object Detection** using YOLOv3
3. **Instance Segmentation** using basic depth-based methods and YOLOv11x-seg

---

## Environment Setup

### Prerequisites
- Linux (tested on Ubuntu 20.04/22.04)
- Python 3.9+
- CUDA (optional, for GPU acceleration with YOLO models)

### Dependencies
Core packages:
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `scipy`
- `ultralytics` (for YOLOv11)
- `torch`, `torchvision` (for YOLO models)

### Installation
If using conda/mamba (recommended):
```bash
# Activate your environment
conda activate AER1515  # or your environment name

# Install core dependencies
pip install numpy opencv-python matplotlib scipy pandas

# Install YOLO dependencies
pip install ultralytics torch torchvision
```

For CPU-only PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Structure

```
assignment3/
├── data/
│   ├── train/
│   │   ├── left/           # Left camera images
│   │   ├── disparity/      # Disparity maps (uint16 PNG, scale=256)
│   │   ├── calib/          # Camera calibration files
│   │   ├── gt_depth/       # Ground truth depth maps
│   │   ├── gt_labels/      # KITTI format object labels (bbox + 3D info)
│   │   ├── gt_segmentation/# Ground truth instance segmentation masks
│   │   ├── est_depth/      # [OUTPUT] Estimated depth maps
│   │   └── est_segmentation/# [OUTPUT] Estimated segmentation masks
│   └── test/
│       ├── left/
│       ├── disparity/
│       ├── calib/
│       ├── est_depth/      # [OUTPUT]
│       └── est_segmentation/# [OUTPUT]
├── yolo/
│   ├── yolo11n-seg.pt      # YOLOv11 nano segmentation model
│   ├── yolo11x-seg.pt      # YOLOv11 extra-large segmentation model
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
└── figures/                # [OUTPUT] Generated plots and visualizations
```

---

## Part 1 — Dense Depth Estimation

### Description
Converts stereo disparity maps to dense depth maps using the formula:
$Z = \frac{f \times B}{d}$
where:
- \( Z \) = depth in meters
- \( f \) = focal length (from calibration)
- \( B \) = baseline (from calibration)
- \( d \) = disparity in pixels

### Script
`part1_estimate_depth.py`

### Usage
**Train set:**
```bash
python part1_estimate_depth.py --dataset train
```

**Test set:**
```bash
python part1_estimate_depth.py --dataset test
```

### Outputs
- Depth maps saved as 16-bit PNG in `data/{train,test}/est_depth/`
- Encoding: `uint16_value = depth_in_meters × 256`
- Depths > 80m and < 10 cm are clipped to 0
- For test set: generates a stacked visualization `est_depth_stack.png` (5×1 subplot)

### Evaluation (Train only)
Compare estimated depth against ground truth:
```bash
python tools/evaluate_est_depth.py \
  --gt_dir data/train/gt_depth \
  --est_dir data/train/est_depth \
  --out_csv figures/depth_metrics.csv
```

**Metrics computed:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- AbsRel (Mean Absolute Relative Error)

---

## Part 2 — Object Detection with YOLO

### Description
Detects objects (cars, pedestrians, etc.) in images using YOLOv3. Applies Non-Maximum Suppression (NMS) to filter overlapping detections.

### Script
`part2_yolo.py`

### Usage
**Train set:**
```bash
python part2_yolo.py --dataset train
```

**Test set:**
```bash
python part2_yolo.py --dataset test
```

### Key Parameters (in script)
- `confidence_th = 0.51`: Minimum confidence to keep a detection
- `threshold = 0.4`: NMS IoU threshold (0.3–0.7 typical)

### Outputs
- For test set: saves a stacked visualization `yolo_detections_stack.png` (5×1 subplot) to `figures/`
- Bounding boxes and labels drawn on images

### Tuning
- Adjust `confidence_th` to filter weak detections
- Adjust `threshold` to control overlap suppression

---

## Part 3 — Instance Segmentation

### 3.1 Basic Depth-Based Method

#### Description
Simple segmentation using depth proximity:
1. For each object bbox (from GT labels), compute average depth from the depth map
2. Segment pixels within the bbox that are within a threshold distance from the average depth
3. Uses GT depth and GT bounding boxes (train only)

#### Script
`part3_segmentation_basic.py`

#### Usage
```bash
python part3_segmentation_basic.py --dataset train --depth-threshold 5.8
```

**Key Parameters:**
- `--depth-threshold`: Distance threshold in meters (can be tuned for best performance)

#### Outputs
- Binary segmentation masks in `data/train/est_segmentation/`
- Encoding: 0 = foreground (object), 255 = background

#### Evaluation
```bash
python tools/evaluate_segmentation.py \
  --est_dir data/train/est_segmentation \
  --gt_dir data/train/gt_segmentation \
  --out_csv figures/segmentation_metrics.csv
```

**Metrics computed:**
- IoU (Intersection over Union)
- Pixel Accuracy
- Precision / Recall
---

### 3.2 YOLOv11x-seg Method

#### Description
Uses a pre-trained YOLOv11x-seg model (62.1M parameters) for instance segmentation. Filters results to only include "car" class masks.

#### Script
`part3_segmentation_yolov11.py`

#### Usage
**Train set:**
```bash
python part3_segmentation_yolov11.py --dataset train
```

**Test set:**
```bash
python part3_segmentation_yolov11.py --dataset test
```

#### Outputs
- Binary segmentation masks in `data/{train,test}/est_segmentation/`
- Encoding: 0 = car (foreground), 255 = background

#### Notes
- Model checkpoint: `yolo11x-seg.pt` will be download automatically via ultralytics if missing
- Uses `retina_masks=True` for original resolution mask outputs
---

## Utility Tools

### 3.3 Stack Images Tool

Creates vertical subplots of all images in a directory (used for reports).

#### Script
`tools/stack_images.py`

#### Usage
```bash
python tools/stack_images.py \
  --input_dir data/train/est_segmentation \
  --output_path figures/segmentation_stack.png
```

**Options:**
- `--extension png`: Image format to look for (default: png)

---
## Workflow Summary

### Complete Pipeline (Train)
```bash
# 1. Generate depth maps
python part1_estimate_depth.py --dataset train

# 2. Evaluate depth
python tools/evaluate_est_depth.py \
  --gt_dir data/train/gt_depth \
  --est_dir data/train/est_depth

# 3. Detect objects
python part2_yolo.py --dataset train

# 4. Generate segmentation (basic method)
python part3_segmentation_basic.py --dataset train --depth-threshold 5.8

# 5. Evaluate segmentation
python tools/evaluate_segmentation.py

# 6. Generate segmentation (YOLO method)
python part3_segmentation_yolov11.py --dataset train

# 7. Create visualizations
python tools/stack_images.py --input_dir data/train/est_segmentation --output_path figures/seg_stack.png
```

### Complete Pipeline (Test)
```bash
# 1. Generate depth maps
python part1_estimate_depth.py --dataset test

# 2. Detect objects
python part2_yolo.py --dataset test

# 3. Generate segmentation (YOLO only, no GT available)
python part3_segmentation_yolov11.py --dataset test
```

---

## References

- **KITTI Dataset:** [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- **YOLOv3:** Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018
- **YOLOv11:** Ultralytics, [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

---