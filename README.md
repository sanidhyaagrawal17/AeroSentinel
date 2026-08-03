# AeroSentinel: Aerospace Defect Detection
**Team Cygnus**

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
Our current best model is a YOLO11 Nano architecture (`yolo11n_aws_best.pt`), which significantly outperformed YOLOv8n. 
**Overall Validation mAP50: 61.7%**

**Per-Class Breakdown:**
- Patches: **91.7%**
- Inclusion: **71.7%**
- Scratches: **65.1%**
- Rolled-in Scale: **48.0%**
- Pitted Surface: **47.3%**
- Crazing: **46.6%**

### Experimental Rigor
This baseline was achieved by utilizing GA-tuned hyperparameters on strictly raw data. To ensure this was the optimal path, we extensively tested and formally ruled out several other approaches:
- **Crazing Defect Bakeoff**: We attempted three focused interventions to fix the low Crazing AP (Class Balancing, Negative Sampling, and Synthetic Augmentation). All three severely degraded performance compared to the raw YOLO11n baseline.
- **Model Scaling**: Larger YOLOv8s and YOLOv8m architectures yielded no meaningful improvements over Nano, proving capacity was not the bottleneck.
- **Custom Attention Modules**: Injecting RFAConv modules added computational overhead without improving spatial context for complex textures.

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
- **Texture Defect Struggles**: High-frequency, subtle texture defects like *Crazing* (46.6% mAP) remain inherently difficult for standard convolutional architectures to distinguish from natural surface noise.
- **Target Accuracy**: We have not yet reached our ambitious 77% mAP50 goal, though we established a rigorously verified 61.7% baseline.

## Dataset Attribution
This project utilizes the NEU-DET (Northeastern University Surface Defect Database) dataset for training and validation. NEU-DET is a public research benchmark dataset widely used for surface defect detection tasks. We acknowledge and thank the original researchers for providing this dataset.
