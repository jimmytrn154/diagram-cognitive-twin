# Project Progress Tracker

## Project

**Cognitive Twin-Ready Engineering Diagram Understanding Using SAM, CLIP, and Graph Extraction**

**Duration:** 8 weeks  
**Target workload:** 10–12 hours per week

---

## Current Status

**Current phase:** Week 1 execution

**Current primary dataset:** Kaggle P&ID Symbols dataset

**Current focus:**

```text
P&ID tile
   |
   v
SAM candidate extraction
   |
   v
Candidate filtering
   |
   v
Comparison with YOLO ground truth
```

---

# Week 1 — Setup and Dataset Understanding

## Goal

Prepare the environment, understand the dataset, verify annotations and splits, and successfully run SAM on at least one image.

## Tasks

### Repository and Environment

- [x] Create Git repository
- [x] Create Python virtual environment (`messi`)
- [x] Install PyTorch
- [x] Install OpenCV
- [x] Install NumPy, Pillow, Matplotlib, and Jupyter
- [x] Install SAM
- [x] Verify GPU / CUDA availability
- [x] Create initial repository folder structure
- [x] Add `requirements.txt`
- [x] Add `README.md`

Current environment notes:

- Active project virtual environment: `messi`
- Installed baseline dependencies from `requirements.txt` are available in `messi`
- `segment-anything` is installed and a SAM `vit_l` checkpoint is available locally
- PyTorch in `messi` is CPU-only (`torch 2.13.0+cpu`), so CUDA is currently unavailable
- Repository scaffold now includes `data/`, `notebooks/`, `src/`, `outputs/`, `ontology/`, and `app/`
- Week 1 work can proceed on CPU for dataset exploration and small-scope SAM tests

### Dataset Setup

- [x] Download P&ID Symbols dataset
- [x] Inspect example image tiles
- [x] Inspect corresponding YOLO annotation file
- [x] Confirm annotation format:
  - `class_id`
  - `x_center`
  - `y_center`
  - `width`
  - `height`
- [x] Confirm that coordinates are normalized
- [x] Observe filename pattern such as `9_640_640.jpg`
- [x] Confirm exact tile naming convention
- [x] Confirm image tile size across dataset
- [x] Check train/validation/test split files
- [x] Verify whether original diagram IDs overlap across splits
- [x] Count images per split
- [ ] Count annotations per class
- [ ] Map class IDs to class names

### Dataset Exploration Notebook

File:

```text
notebooks/01_data_exploration.ipynb
```

Tasks:

- [x] Load one image and its `.txt` annotation
- [x] Convert normalized YOLO coordinates to pixel coordinates
- [x] Draw ground-truth bounding boxes
- [ ] Display class names on bounding boxes
- [ ] Visualize 10–20 example tiles
- [ ] Record image dimensions
- [ ] Analyze symbol size distribution
- [ ] Analyze class distribution
- [ ] Identify simple, medium, and complex examples
- [ ] Select a small development subset

Notebook run summary:

- Executed successfully in the `messi` kernel (`Python 3.10.10`)
- Dataset paths resolved correctly from `C:\Users\jimmy\Downloads\pidset`
- Split counts confirmed: 27,000 train images and 3,000 val images
- Example annotation check completed on `9_640_640.jpg`
- Example tile shape confirmed: `(1280, 1280, 3)`
- Example annotation count confirmed: `7`
- The notebook rendered a 12-tile annotated visualization grid

### Initial SAM Test

- [ ] Load SAM model
- [ ] Run SAM on one original tile
- [ ] Generate masks
- [ ] Visualize all raw masks
- [ ] Save one raw-mask overlay
- [ ] Record approximate number of masks produced
- [ ] Write initial observations

## Week 1 Deliverables

- [x] Working Python environment
- [ ] Dataset organized
- [x] Dataset split verified
- [x] Annotation visualization working
- [x] `01_data_exploration.ipynb`
- [ ] Selected development subset
- [ ] SAM successfully runs on one tile
- [x] Initial observations documented

## Week 1 Notes

### Dataset observations

- The dataset contains tiled P&ID images.
- Example tile size observed: 1280 × 1280 pixels.
- Filenames appear to encode original diagram ID and crop position.
- Tiles may overlap.
- Each image has a YOLO-format label file.
- Ground-truth annotations can be used to evaluate SAM candidate extraction.
- Dataset root used in the notebook: `C:\Users\jimmy\Downloads\pidset`
- Split files found: `train (2).txt` and `val (1).txt`
- Split sizes confirmed: 27,000 train images and 3,000 val images.
- Diagram-ID overlap is substantial: 500 unique train diagram IDs, 498 unique val diagram IDs, and 498 overlapping IDs.

### Questions to resolve

- [x] What exactly do the filename coordinates represent?
- [ ] What is the overlap ratio between adjacent tiles?
- [x] Are tiles from the same original diagram kept in the same dataset split?
- [ ] Which class IDs correspond to which engineering symbols?

---

# Week 2 — Preprocessing and SAM Candidate Extraction

## Goal

Build and evaluate the pipeline:

```text
Input image
   |
   v
Preprocessing
   |
   v
SAM automatic mask generation
   |
   v
Raw masks
   |
   v
Candidate feature extraction
   |
   v
Geometric filtering
   |
   v
Candidate crops
   |
   v
Ground-truth comparison
```

---

## 2.1 Preprocessing Experiments

Test the following variants:

- [ ] Original RGB
- [ ] Grayscale
- [ ] Binary thresholding
- [ ] Denoising
- [ ] Optional contrast enhancement

For each variant:

- [ ] Run SAM
- [ ] Count generated masks
- [ ] Compare candidate coverage
- [ ] Record whether preprocessing improves or hurts segmentation

### Decision

Best preprocessing configuration:

```text
TBD
```

Reason:

```text
TBD
```

---

## 2.2 Raw SAM Mask Generation

Notebook:

```text
notebooks/02_sam_segmentation.ipynb
```

Tasks:

- [ ] Run automatic mask generation
- [ ] Save raw mask metadata
- [ ] Save segmentation overlays
- [ ] Run on at least 10 development images
- [ ] Record masks per image
- [ ] Inspect duplicate masks
- [ ] Inspect fragmented symbols
- [ ] Inspect page/background masks
- [ ] Inspect text masks
- [ ] Inspect line masks

---

## 2.3 Candidate Metadata

For each SAM mask, store:

- [ ] candidate ID
- [ ] source image
- [ ] bounding box
- [ ] mask area
- [ ] width
- [ ] height
- [ ] aspect ratio
- [ ] SAM predicted IoU if available
- [ ] SAM stability score if available
- [ ] crop path

Suggested structure:

```json
{
  "candidate_id": "candidate_001",
  "source_image": "9_640_640.jpg",
  "bbox": [100, 200, 170, 260],
  "width": 70,
  "height": 60,
  "area": 2400,
  "aspect_ratio": 1.17,
  "sam_score": 0.91,
  "crop_path": "outputs/crops/9_640_640/candidate_001.png"
}
```

---

## 2.4 Candidate Filtering

Initial rules to test:

### Small-area filtering

- [ ] Remove extremely small regions
- [ ] Record symbols accidentally removed

### Large-area filtering

- [ ] Remove page-level or very large regions
- [ ] Record symbols accidentally removed

### Aspect-ratio filtering

- [ ] Identify long thin regions
- [ ] Separate likely connectors from compact symbol candidates

Suggested conceptual routing:

```text
SAM mask
   |
   v
Geometric analysis
   |
   +--> compact region ------> symbol candidate
   |
   +--> long thin region ----> possible connector
   |
   +--> tiny region ---------> noise / text candidate
   |
   `--> huge region ---------> discard
```

Do not permanently discard uncertain candidates until recall has been evaluated.

---

## 2.5 Ground-Truth Matching

Convert each SAM mask into a bounding box.

Compare SAM candidate boxes with YOLO ground-truth boxes.

Possible matching methods:

- [ ] IoU threshold
- [ ] Ground-truth center inside candidate box
- [ ] Candidate center inside ground-truth box
- [ ] Combined overlap criterion

Initial IoU thresholds to test:

- [ ] 0.3
- [ ] 0.5

### Main metric

**Candidate Recall**

```text
number of ground-truth symbols matched by at least one valid SAM candidate
---------------------------------------------------------------------------
total number of ground-truth symbols
```

Track:

- [ ] raw SAM candidate recall
- [ ] filtered candidate recall
- [ ] average candidates per image
- [ ] average candidates per ground-truth object
- [ ] missed symbol classes

---

## 2.6 Failure Analysis

### Failure Type 1 — Text Over-Segmentation

Example:

```text
P-101
 |
 v
P
-
1
0
1
```

Status:

- [ ] Observed
- [ ] Documented

Possible mitigation:

- area filtering,
- OCR-specific handling later.

---

### Failure Type 2 — Symbol Fragmentation

Example:

```text
One valve
   |
   v
3 SAM masks
```

Status:

- [ ] Observed
- [ ] Documented

Possible mitigation:

- merge nearby masks,
- merge overlapping bounding boxes,
- retain a larger enclosing candidate.

---

### Failure Type 3 — Pipeline Segmentation

Example:

```text
Long pipe -> SAM candidate
```

Status:

- [ ] Observed
- [ ] Documented

Possible mitigation:

- aspect-ratio filtering,
- move to connector candidate pool.

---

### Failure Type 4 — Multiple Objects in One Mask

Status:

- [ ] Observed
- [ ] Documented

Possible mitigation:

- size filtering,
- alternative SAM parameters,
- prompted segmentation if necessary.

---

### Failure Type 5 — Missed Symbol

Status:

- [ ] Observed
- [ ] Documented

Possible causes:

- symbol too small,
- symbol connected to lines,
- symbol visually ambiguous,
- insufficient contrast.

---

## Week 2 Deliverables

- [ ] `02_sam_segmentation.ipynb`
- [ ] preprocessing comparison
- [ ] raw SAM masks
- [ ] mask overlays
- [ ] candidate metadata JSON
- [ ] filtered candidate crops
- [ ] ground-truth matching script
- [ ] raw candidate recall result
- [ ] filtered candidate recall result
- [ ] failure-case report
- [ ] Week 3-ready candidate crop dataset

---

## Week 2 Success Criteria

The Week 2 pipeline should be able to run:

```text
P&ID image
   |
   v
SAM
   |
   v
Raw candidates
   |
   v
Filtering
   |
   v
Candidate crops + metadata
```

A successful Week 2 result should answer:

1. How many annotated symbols can SAM recover?
2. Which classes are commonly missed?
3. How many unnecessary masks are generated?
4. Which filtering rules are useful?
5. How much recall is lost after filtering?
6. Are the surviving crops suitable for CLIP?

---

# Week 3 — CLIP Classification

## Goal

Classify SAM-generated candidate crops.

### Planned tasks

- [ ] Define initial class list
- [ ] Build prototype crop library using training annotations
- [ ] Extract CLIP embeddings
- [ ] Test zero-shot text prompts
- [ ] Test prototype-image retrieval
- [ ] Compare both methods
- [ ] Evaluate on validation images
- [ ] Report top-1 and top-3 accuracy

### Important split rule

Do not use overlapping crops of the same original symbol for both prototype creation and evaluation.

---

# Week 4 — OCR and Text Association

- [ ] Select OCR engine
- [ ] Detect text regions
- [ ] Recognize text
- [ ] Clean OCR results
- [ ] Associate labels with nearby components
- [ ] Evaluate equipment-tag extraction
- [ ] Document OCR failure cases

---

# Week 5 — Connector and Line Extraction

- [ ] Remove or mask detected components
- [ ] Remove or mask text regions
- [ ] Extract remaining line network
- [ ] Apply skeletonization
- [ ] Detect endpoints
- [ ] Detect junctions
- [ ] Associate endpoints with components
- [ ] Detect arrows where possible
- [ ] Generate preliminary graph edges

---

# Week 6 — Graph Construction and Digital-Twin Mapping

- [ ] Define graph schema
- [ ] Create component nodes
- [ ] Create connection edges
- [ ] Attach labels
- [ ] Attach confidence scores
- [ ] Export JSON
- [ ] Export NetworkX graph
- [ ] Define digital-twin entity mappings
- [ ] Optionally export RDF triples

---

# Week 7 — Query and Review Interface

- [ ] Create Streamlit or notebook interface
- [ ] Show source diagram
- [ ] Show detected candidates
- [ ] Show classified components
- [ ] Show OCR labels
- [ ] Show extracted graph
- [ ] Flag low-confidence objects
- [ ] Add example graph queries
- [ ] Define human correction workflow

---

# Week 8 — Final Demo and Report

## Final examples

- [ ] P&ID/PFD success case
- [ ] Simulink-style example
- [ ] Uncertainty / failure case

## Final deliverables

- [ ] Working prototype
- [ ] Final report
- [ ] 10–15 slide presentation
- [ ] Clean repository
- [ ] Final README
- [ ] Example JSON graph output
- [ ] Demo screenshots / video
- [ ] Limitations section
- [ ] Future work section

---

# Experiment Log

Use this section to record each experiment.

## Experiment Template

### Experiment ID

```text
EXP-XXX
```

### Date

```text
YYYY-MM-DD
```

### Objective

```text
What question is this experiment answering?
```

### Configuration

```text
Model:
Input:
Preprocessing:
Parameters:
Dataset subset:
```

### Results

```text
Raw masks:
Filtered masks:
Ground-truth objects:
Matched objects:
Candidate recall:
```

### Observations

```text
-
-
-
```

### Decision

```text
What should be changed or kept for the next experiment?
```

---

# Key Decisions

| Date | Decision | Reason |
|---|---|---|
| TBD | Use P&ID Symbols dataset for initial SAM experiments | Provides tiled images and YOLO ground truth |
| TBD | Treat SAM as a class-agnostic candidate generator | Semantic classification is handled by CLIP |
| TBD | Prioritize candidate recall during Week 2 | False candidates can be filtered later |
| TBD | Use provided dataset split initially | Dataset already includes predefined splits |

---

# Open Questions

- [ ] Does the provided dataset split prevent overlap between original diagrams?
- [ ] What are the exact class ID-to-name mappings?
- [ ] Which SAM variant should be used for the main experiment?
- [ ] Which SAM automatic mask generator parameters work best?
- [ ] What IoU threshold should define a successful candidate match?
- [ ] Should text-like masks be filtered immediately or retained for OCR experiments?
- [ ] Should connector-like masks be preserved separately?
- [ ] How many CLIP classes should be used initially?
- [ ] Should fine-grained valve classes be merged into broader categories first?

---

# Current Next Actions

1. [x] Create `01_data_exploration.ipynb`
2. [x] Load the predefined dataset split
3. [ ] Map class IDs to class names
4. [x] Check diagram-ID overlap across splits
5. [ ] Visualize annotations on 10–20 tiles
6. [ ] Select a development subset
7. [x] Install and load SAM
8. [ ] Run the first SAM segmentation experiment
9. [ ] Save all raw masks and overlays
10. [ ] Start recording results in the Experiment Log
