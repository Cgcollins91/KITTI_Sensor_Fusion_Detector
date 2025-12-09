# KITTI 3D Object Detection Pipeline

This repository provides a full LiDAR + RGB processing pipeline for the KITTI dataset, including 2D detection loading, 3D projection, frustum culling, clustering, and 3D bounding box estimation.

This forked repository includes a torch pipeline to train a car detector.

Paths to data files for training and input video will need to be edited.

This README file is in work. The Main files are training.py, read_video.py, and pipeline_with_detector.py

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone git@github.com:Cgcollins91/KITTI_Sensor_Fusion_Detector.git
cd KITTI_Sensor_Fusion_Detector
```

### 2. Create Conda Environment

This project includes an `environment.yml` for reproducible setup.

```bash
conda env create -f environment.yml
conda activate carla
```

---

## Dataset Setup

Download the KITTI **object detection dataset**:

* Images (`image_2/`)
* Velodyne LiDAR scans (`velodyne/`)
* Calibration files (`calib/`)
* Labels (`label_2/`)

This dataset is used to train our model

## Project Structure

```
ENPM818Z_FALL_2025_RWA_1/
├── pipeline_with_detector.py   # jupyter cell version of intermediate steps of loading Kitti dataset image, labels, and bounding boxes
├── starter.py                  # KITTI loaders for image + lidar
├── detector.py                 # KITTI label loading
├── environment.yml             # Conda setup
├── training.py                 # Model Training Pipeline on Kitti Dataset
├── read_video.py               # Run model on input video, outputs video with bounding boxes around detected cars
└── README.md                   # Documentation
```


## Output

Running the pipeline produces:

* RGB + LiDAR projection
* Depth visualization
* 2D detection overlays
* 3D LiDAR clusters
* Axis-aligned / oriented bounding boxes

Running training.py outputs:
- model.pth trained car    # detection model

Running read_video.py outputs:
output_detector.mp4        # input video with added bounding boxes for detected carThins

---

## Dependencies

* Python 3.9+
* NumPy
* OpenCV
* Open3D
* Matplotlib
---
