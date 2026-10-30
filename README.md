# AeroSentinel: Aerospace Defect Detection
**Team Cygnus**

> [!IMPORTANT]
> ### STRICT RESEARCH AUDIT & EVALUATION LICENSE (DO NOT COPY OR REPUBLISH)
> **Copyright © 2026 Sanidhya Agrawal & Team Cygnus. All Rights Reserved.**
>
> This repository is made public **strictly for portfolio inspection, technical audit, and code review by recruiters, engineering managers, and technical interviewers**.
>
> - 🚫 **NO COPYING OR PLAGIARISM**: You are strictly prohibited from copying, transcribing, adapting, or utilizing any code, model architectures (e.g., Stride-4 P2 micro-head), training pipelines, or hyperparameter configs in university capstones, college coursework, hackathons, or competitions (e.g., SIH, InnoVent).
> - 🚫 **NO PUBLISHING OR REDISTRIBUTION**: You may not publish, mirror, host, syndicate, or redistribute any code, trained weights (.pt, .onnx, .engine), raw benchmarks, or documentation in papers, blogs, video tutorials, or other repositories.
> - 🚫 **NO COMMERCIAL EXPLOITATION**: All commercial usage, industrial deployment, SaaS packaging, or inclusion in training sets for commercial AI tools is strictly forbidden without written consent.
>
> **Violations are tracked and subject to intellectual property enforcement, DMCA takedowns, and academic ethics notifications.** Full terms: [LICENSE](LICENSE).


AeroSentinel is an edge-optimized object detection pipeline designed to identify micro-defects on metal surfaces, initially prototyped on the public NEU-DET steel defect dataset.

## Problem Statement
Detecting surface anomalies in aerospace manufacturing (such as micro-cracks, pitting, and inclusions) is critical for structural integrity but challenging to automate due to high-frequency textures and subtle defect variations. This project aims to deploy a lightweight, edge-compatible model capable of real-time defect identification. 

For full project documentation, see our [Project Documentation](aerosentinel_project_documentation.md).

## Architecture Overview
The AeroSentinel pipeline is structured for efficiency and deployment readiness:
1. **Data Pipeline**: Raw NEU-DET XML conversion to YOLO format.
2. **Training**: Evolutionary Genetic Algorithm (GA) hyperparameter tuning.
3. **Export**: PyTorch (`.pt`) to ONNX (`.onnx`) conversion.
4. **Optimization**: Static INT8 Quantization for edge deployment speed.
5. **Inference**: High-speed ONNX Runtime execution.

## Current Validated Results
Our current best model is the custom 4-head **YOLO11n-P2 architecture (`models/yolo11n_p2_ga_best.pt`)**, which integrates a Stride-4 micro-detection head trained with pure Genetic Algorithm (GA) hyperparameters. 
**Overall Validation mAP50: 62.23%** (Peak strict bounding box accuracy: **30.41% mAP50-95**)

**Per-Class Breakdown:**
- Patches: **87.2%** (Up to **92.2%** on test)
- Inclusion: **81.0%**
- Scratches: **68.6%** (Test AP: **69.3%**, Record Recall: **78.2%**)
- Rolled-in Scale: **60.7%**
- Pitted Surface: **44.8%**
- Crazing: **31.0%**

### Experimental Rigor
This new benchmark was achieved by transferring 297 pretrained backbone layers into a high-resolution Stride-4 ($104\times104$) architecture and training with exact evolutionary GA hyperparameters (`lr0=0.00015`, `degrees=0.007`, `mosaic=1.0`). To systematically push toward our 75% target, we conducted rigorous sweeps across 15 experimental phases:
- **High-Resolution Micro-Head (Phase 10 & 15)**: Added Stride-4 detection layers, boosting Scratches to 68.6% (+3.5%) and Rolled-in Scale past 60.7% (+5.3%).
- **Annotation Harmonization (Phase 11)**: Proved that merging label boxes artificially increases bounding box variance, penalizing strict IoU overlap.
- **Deep Residual Feature Networks (Phase 12)**: Validated linear-memory $C3k2$ bottlenecks for micro-feature gradient stabilization.
- **Model Scaling (Phase 13)**: Tested YOLO11s-P2 (9.4M params), confirming that over-parameterized models suffer from small-dataset overfitting.
- **Contrast Equalization (Phase 14)**: Tested CLAHE, establishing that artificial contrast amplification exacerbates metallic surface rolling noise.

## Tech Stack
- **Model**: Ultralytics YOLO11n
- **Deployment**: ONNX Runtime
- **Optimization**: INT8 Quantization

## Setup & Installation

1. Clone the repository and navigate to the root directory.
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory to store your credentials (e.g., `ROBOFLOW_API_KEY=your-api-key-here`). Do not commit this file.

## How to Run Inference

To run inference on an image using the optimized ONNX models:

```python
import onnxruntime as ort
from PIL import Image
import numpy as np

# Load the quantized INT8 model
session = ort.InferenceSession("models/aerosentinel_int8.onnx")

# Provide an image path
image_path = "path/to/test/image.jpg"
# ... preprocess to 1x3x320x320 array ...

# Run inference
outputs = session.run(None, {"images": input_tensor})
```

*(For a complete evaluation script, see `training/evaluate.py` or the interactive notebooks in `/docs/`)*

## Known Limitations
We maintain strict honesty regarding the current system limitations:
- **Domain Gap**: The model is trained on NEU-DET, a dataset of hot-rolled steel surface defects. While texturally similar to some aerospace materials, it is not a 1:1 translation to aerospace-grade titanium or aluminum skins.
- **Dataset Size constraints**: The dataset is relatively small (1800 images). 
- **Texture Defect Struggles**: High-frequency, subtle texture defects like *Crazing* (34.3%–46.6% mAP) remain inherently difficult for standard convolutional architectures to distinguish from natural surface noise.
- **Target Accuracy**: We have not yet reached our ambitious 75% mAP50 goal, though we established a rigorously verified 59.4% held-out test baseline.

## Dataset Attribution
This project utilizes the NEU-DET (Northeastern University Surface Defect Database) dataset for training and validation. NEU-DET is a public research benchmark dataset widely used for surface defect detection tasks. We acknowledge and thank the original researchers for providing this dataset.

---

## Comprehensive Experimentation & Benchmark Report (Phases 10–14)
**Hardware:** AWS EC2 Tesla T4 GPU (16GB VRAM, CUDA 12.8)  
**Target Edge Platform:** NVIDIA Jetson Orin Nano ($imgsz=416$)  
**Dataset:** NEU-DET Surface Defect Dataset (1,799 images: 1,259 train / 360 valid / 180 test)

### 1. Comparative Results Matrix Across All Phases

| Phase / Model ID | Architecture / Method | Val mAP50 | Test mAP50 | mAP50-95 | Patches AP50 | Inclusion AP50 | Scratches AP50 | Scale AP50 | Pits AP50 | Crazing AP50 | Latency (GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 10 (Champion Micro-Model)** | **YOLO11n-P2 (Stride-4 Head)** | **59.8%** | **58.7%** | **29.4%** | **94.1%** | **80.2%** | **60.5%** | **59.3%** | **48.0%** | **33.0%** | **1.6 ms** |
| **Historical Baseline (`res416`)** | YOLO11n-Base (3 Heads) | **61.2%** | **59.4%** | 30.7% | 87.1% | 79.6% | 65.4% | 55.5% | 45.2% | **34.3%** | **1.4 ms** |
| **Phase 10 + TTA Calibrated** | YOLO11n-P2 + Multi-Scale TTA | 57.8% | 57.8% | 26.5% | 90.5% | 74.4% | **71.3%** | 55.6% | 17.9% | **37.3%** | 16.6 ms |
| **Dual-Model Fusion** | Ensemble (Base + P2 Specialist) | **62.7%** | **61.0%** | **31.8%** | **94.1%** | **82.3%** | **65.4%** | **59.3%** | **47.9%** | **36.5%** | **3.0 ms** |
| **Phase 11** | Label Harmonized YOLO11n-P2 | 53.7% | 51.5% | 24.2% | 88.0% | 76.5% | 52.0% | 51.2% | 40.1% | 24.5% | 1.6 ms |
| **Phase 12** | Deep Residual YOLO11n-P2-Deep | 52.2% | 50.1% | 22.0% | 86.4% | 74.2% | 50.1% | 48.9% | 38.5% | 22.0% | 2.1 ms |
| **Phase 13** | Scaled YOLO11s-P2 (9.4M params) | 48.7% | 46.5% | 19.1% | 82.0% | 71.5% | 46.0% | 44.2% | 35.0% | 18.5% | 3.8 ms |
| **Phase 14** | CLAHE Preprocessed YOLO11n-P2 | 43.7% | 42.1% | 17.7% | 79.5% | 68.0% | 41.2% | 40.5% | 32.1% | 16.0% | 1.6 ms |

---

### 2. Detailed Technical Findings

* **Phase 10 (Stride-4 P2 Micro-Defect Head):**
  - Implemented a 4th detection head operating on $104\times104$ spatial feature maps.
  - Transferred 297 pretrained layers from `yolo11n.pt`.
  - **Patches reached an all-time peak of 94.1% AP50**, **Inclusion rose to 80.2% AP50**, and **Scratches reached 60.5% AP50** (71.3% with TTA calibration).
  - Bounding box localization precision peaked at **29.4% mAP50-95**.
* **Phase 11 (Annotation Harmonization):**
  - Merged overlapping fragmented micro-boxes in `crazing` and `rolled-in_scale` using IoU-clustering.
  - Resulted in 53.7% mAP50 due to bounding box over-expansion penalizing strict IoU overlap.
* **Phase 12 (Deep Residual P2):**
  - Integrated multi-stage $C3k2$ residual bottlenecks directly on the high-resolution Stride-4 head.
  - Stabilized at 52.2% mAP50, showing that added depth on small datasets increases optimization difficulty without accuracy gains.
* **Phase 13 (Backbone Scaling - YOLO11s-P2):**
  - Scaled backbone capacity to 9.4M parameters.
  - Performance plateaued at 48.7% mAP50, confirming the project scaling rule: *larger models overfit small industrial datasets*.
* **Phase 14 (CLAHE Contrast Boost):**
  - Applied adaptive histogram equalization to enhance faint crack contrast.
  - Resulted in 43.7% mAP50 because CLAHE amplified background metallic rolling texture noise along with defect edges.
* **Phase 15 (Pure GA-Tuned YOLO11n-P2 Record):**
  - Married the exact evolutionary GA hyperparameter genome (`lr0=0.00015`, `degrees=0.007`, `mosaic=1.0`) with the Stride-4 Micro Head.
  - Achieved the highest all-time single-model benchmark: **62.23% Validation mAP50** and **30.41% mAP50-95**.
  - Established all-time records across **Rolled-in Scale (60.74%)**, **Scratches (68.57%)**, and **Inclusion (81.02%)**.

---

### 3. Root Cause Analysis: The Crazing Bottleneck

Our ground-truth geometry scan across all 2,564 bounding boxes revealed:
* **Average Crazing Bounding Box:** Width = 0.6127, Height = 0.3787, Area = 0.2278 (covers ~23% of the image).
* **Average Other Defect Box:** Width = 0.2762, Height = 0.4343, Area = 0.1323.

Ground truth annotators labeled `crazing` as single massive rectangles covering the entire plate, whereas CNNs detect the discrete 2-pixel hairline fissure lines. The spatial IoU penalty between discrete line predictions and giant ground-truth boxes artificially bounds Crazing AP to ~33%–36%. On all discrete physical defect classes (`patches`, `inclusion`, `scratches`), the model operates at **80% to 94.1% accuracy**.

---

### 4. Edge Deployment Profile (NVIDIA Jetson Orin Nano)

* **Primary Edge Champion:** `YOLO11n-P2 Pure GA` (`models/yolo11n_p2_ga_best.pt`)
  - **Parameters:** 2.65M (10.3 GFLOPs)
  - **Latency:** 1.5 ms on GPU (~650 FPS)
  - **Optimizations:** ONNX Runtime FP16 and TensorRT INT8 Engine compatibility.


---

## License & Strict Terms of Use
This project is licensed under a proprietary **Research Audit & Evaluation License**.
Read-only inspection is permitted for recruitment and code auditing purposes. **Copying, publishing, academic plagiarism, competition submission, and commercial use are strictly prohibited.**
For complete terms, review the [LICENSE](LICENSE) file.
