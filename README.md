# Cognitive Twin-Ready Engineering Diagram Understanding

## Overview

This project is building a prototype pipeline that turns engineering diagram tiles into structured, digital-twin-ready outputs.

The current primary path is now **detector-first**, not SAM-first, because the working dataset is already formatted as an **object detection** problem with YOLO annotations.

Current high-level pipeline:

```text
Engineering Diagram Tile
        |
        v
Object Detection Baseline
        |
        v
OCR and Label Association
        |
        v
Connector / Line Extraction
        |
        v
Graph Construction
        |
        v
Digital-Twin-Ready JSON / Knowledge Graph
```

## Current Dataset

Current dataset:

- **P&ID Symbols**  
  https://www.kaggle.com/datasets/hristohristov21/pid-symbols

Key dataset facts already confirmed in the repo work:

- tiled engineering diagram images
- YOLO-format annotation files
- current local root: `C:\Users\jimmy\Downloads\pidset`
- image naming pattern: `<diagram_id>_<crop_x>_<crop_y>.jpg`
- tile size: `1280 x 1280`
- current split files:
  - `train (2).txt`
  - `val (1).txt`
- split sizes:
  - `27,000` train images
  - `3,000` val images

Important caveat:

- the provided train/val split has heavy **diagram-ID overlap**, so it is useful for a baseline but not a strict diagram-isolated generalization benchmark

## Detector-First Baseline

Primary baseline stack:

- **Ultralytics YOLO**
- current working label space: **32 classes**
- comparison tile: `9_640_640.jpg`

The first detector notebook is:

- [notebooks/02_object_detection_baseline.ipynb](./notebooks/02_object_detection_baseline.ipynb)

That notebook is designed to:

1. validate dataset paths and annotations
2. prepare a zero-based detector workspace for Ultralytics
3. run **pre-training inference** on one sample tile
4. train a YOLO baseline
5. validate the resulting model
6. run **post-training inference** on the same sample tile
7. save overlays and prediction-summary JSON files for direct before/after comparison

Detector outputs are written under:

- `outputs/detection/runs/`
- `outputs/detection/overlays/`
- `outputs/detection/reports/`

## Repo Notes

Reusable detector helpers now live under:

- [src/detection](./src/detection)

The older SAM notebook remains in the repo as legacy/reference exploration:

- [notebooks/02_sam_segmentation.ipynb](./notebooks/02_sam_segmentation.ipynb)

It is no longer the primary Week 2 path.
