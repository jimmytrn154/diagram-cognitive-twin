# Cognitive Twin-Ready Engineering Diagram Understanding

## Overview

This project develops a prototype pipeline for converting engineering diagram images into structured, digital-twin-ready graph representations.

The main focus is on P&ID/PFD-style engineering diagrams, with Simulink-style block diagrams treated as a secondary extension.

The proposed pipeline combines:

- **SAM / SAM 2** for class-agnostic candidate region extraction
- **CLIP / OpenCLIP** for semantic symbol classification
- **OCR** for engineering label and tag extraction
- **Classical computer vision** for connector and line extraction
- **Graph construction** for representing components and relationships
- **Digital-twin mapping** for transforming extracted objects into assets, flows, signals, and other machine-readable entities

The project is designed as an 8-week student prototype with an expected workload of approximately 10–12 hours per week.

---

## Project Goal

The target end-to-end workflow is:

```text
Engineering Diagram
        |
        v
Preprocessing
        |
        v
SAM Candidate Extraction
        |
        v
Candidate Filtering
        |
        v
CLIP Symbol Classification
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
        |
        v
Query and Review Interface
```

The minimum viable prototype should accept one clean P&ID/PFD image and produce:

- detected engineering symbols,
- extracted labels,
- approximate connections,
- graph nodes and edges,
- confidence values,
- JSON output.

A stronger version should also process a simple Simulink-style diagram.

---

## Why SAM and CLIP Are Both Used

SAM and CLIP serve different purposes.

### SAM

SAM is used for **class-agnostic candidate extraction**.

It answers:

> Where are the visually meaningful regions?

Typical SAM output:

```text
Mask 1 -> unknown object
Mask 2 -> unknown object
Mask 3 -> unknown object
```

SAM does not normally assign semantic engineering classes such as `Pump`, `Valve`, or `Instrument`.

### CLIP

CLIP is used after SAM to answer:

> What is this candidate object?

Example:

```text
SAM candidate crop -> CLIP -> Gate Valve
SAM candidate crop -> CLIP -> Instrument
SAM candidate crop -> CLIP -> Flow Arrow
```

The connection between the two stages is therefore:

```text
SAM
 |
 v
Candidate masks and bounding boxes
 |
 v
Geometric filtering
 |
 v
Candidate crops
 |
 v
CLIP classification
```

OCR is handled separately because text such as `P-101`, `FT-301`, or equipment tags represents textual identity rather than visual object class.

---

## Current Dataset

The current P&ID dataset is the Kaggle dataset:

**P&ID Symbols**  
https://www.kaggle.com/datasets/hristohristov21/pid-symbols

The dataset contains tiled P&ID images and corresponding YOLO-format annotation files.

Example image filename:

```text
9_640_640.jpg
```

The filename appears to encode:

```text
<original_diagram_id>_<crop_x>_<crop_y>
```

The tiles are 1280 x 1280 pixels and appear to use overlapping crop offsets.

A corresponding YOLO annotation file contains entries in the form:

```text
class_id x_center y_center width height
```

where the coordinates are normalized to the image size.

Example:

```text
32 0.775390625 0.56875 0.09921875 0.0984375
```

The provided dataset split should be used initially. Before classification experiments, it is recommended to verify that overlapping tiles from the same original diagram do not appear across training and validation/test splits.

---

## Current Experimental Focus

The first two weeks focus on determining whether SAM can act as a useful candidate generator for P&ID symbols.

The main experiment is:

```text
P&ID tile
   |
   v
SAM automatic mask generation
   |
   v
Raw candidate masks
   |
   v
Convert masks to bounding boxes
   |
   v
Compare with YOLO ground-truth boxes
   |
   v
Measure candidate recall
   |
   v
Apply geometric filtering
   |
   v
Measure candidate recall again
```

The main research questions are:

1. Can SAM recover annotated engineering symbols without task-specific training?
2. How many unnecessary masks does SAM generate?
3. Can geometric filtering remove obvious false candidates without reducing symbol recall too much?
4. Which symbol types are consistently missed or fragmented?
5. How well do candidate crops support later CLIP classification?

---

## Week 1 Goals

### Environment and Repository Setup

- create the project repository,
- create a Python environment,
- install core dependencies,
- install SAM or SAM 2,
- verify GPU support if available.

### Dataset Understanding

- inspect the provided train/validation structure,
- understand the tile naming convention,
- inspect YOLO annotation files,
- visualize bounding boxes and class IDs,
- check whether original diagram IDs overlap across dataset splits.

### Development Subset

Select approximately 10–20 tiles from different original diagrams for rapid development.

Prefer a mixture of:

- simple tiles,
- medium-complexity tiles,
- dense tiles.

### Initial SAM Test

Run SAM on at least one raw P&ID tile and visualize all generated masks.

---

## Week 2 Goals

### Preprocessing

Test several input variants:

- original RGB image,
- grayscale image,
- thresholded image,
- denoised image.

Do not assume that stronger preprocessing always improves SAM.

### SAM Candidate Extraction

For each image:

1. run automatic mask generation,
2. save mask metadata,
3. convert masks to bounding boxes,
4. visualize all candidate masks.

### Candidate Features

For each candidate, record:

- candidate ID,
- bounding box,
- width,
- height,
- area,
- aspect ratio,
- SAM score,
- stability score if available.

### Candidate Filtering

Initial filtering may use:

- minimum area,
- maximum area,
- aspect ratio,
- page-level mask removal,
- very long/thin region detection.

Long thin candidates may be stored separately as possible connectors rather than discarded completely.

### Evaluation

Compare SAM candidate boxes against the YOLO ground-truth boxes.

The main Week 2 metric is **candidate recall**:

```text
candidate recall =
number of ground-truth symbols matched by at least one usable SAM candidate
/
total number of ground-truth symbols
```

High recall is more important than high precision at this stage.

### Week 2 Outputs

- raw SAM masks,
- candidate metadata,
- filtered candidate crops,
- overlay visualizations,
- candidate recall before filtering,
- candidate recall after filtering,
- documented failure cases.

---

## Planned 8-Week Roadmap

| Week | Focus |
|---|---|
| 1 | Environment setup, dataset inspection, annotation visualization |
| 2 | Preprocessing, SAM candidate extraction, filtering, evaluation |
| 3 | CLIP classification and prototype library |
| 4 | OCR and text-to-symbol association |
| 5 | Connector and line extraction |
| 6 | Graph construction and digital-twin mapping |
| 7 | Query and review interface |
| 8 | Final evaluation, demo, report, and presentation |

---

## Proposed Repository Structure

```text
diagram-cognitive-twin/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- samples/
|   `-- synthetic_simulink/
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_sam_segmentation.ipynb
|   |-- 03_clip_classification.ipynb
|   |-- 04_ocr_text_association.ipynb
|   |-- 05_line_extraction.ipynb
|   |-- 06_graph_construction.ipynb
|   `-- 07_final_demo.ipynb
|
|-- src/
|   |-- preprocessing/
|   |-- segmentation/
|   |-- classification/
|   |-- ocr/
|   |-- connectors/
|   |-- graph/
|   `-- interface/
|
|-- outputs/
|   |-- masks/
|   |-- crops/
|   |-- overlays/
|   |-- graphs/
|   `-- reports/
|
|-- ontology/
|   |-- diagram_dt_schema.md
|   `-- example_triples.ttl
|
|-- app/
|   `-- streamlit_app.py
|
|-- README.md
|-- PROJECT_PROGRESS.md
`-- requirements.txt
```

---

## Expected Final Output

An example graph representation may look like:

```json
{
  "diagram_id": "pid_001",
  "nodes": [
    {
      "id": "Valve_001",
      "type": "Valve",
      "label": "V-203",
      "bbox": [320, 342, 360, 378],
      "confidence": 0.81
    }
  ],
  "edges": [
    {
      "id": "Pipe_001",
      "type": "ProcessFlow",
      "source": "Pump_001",
      "target": "Valve_001",
      "confidence": 0.67
    }
  ]
}
```

---

## Project Scope

This project does **not** aim to solve complete industrial P&ID digitization.

The project should prioritize:

- a clear end-to-end prototype,
- reproducible experiments,
- measurable candidate extraction and classification performance,
- uncertainty-aware outputs,
- human review where predictions are ambiguous.

The system should avoid pretending that every extraction is correct. Low-confidence detections and uncertain graph relationships should be explicitly represented.

---

## Current Status

The project is currently in **Week 1 / Week 2 preparation**.

The P&ID Symbols dataset has been downloaded and inspected. The immediate next step is to build the dataset exploration notebook, visualize the provided annotations, verify the split structure, and begin SAM candidate extraction experiments.
