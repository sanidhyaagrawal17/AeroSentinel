# AeroSentinel

**AI-Powered Offline Structural Vigilance for Aircraft Inspection**

**Team:** Cygnus | **Competition:** TATA AIG 2026 | **Category:** Edge AI

---

## Overview

AeroSentinel is a fully offline, edge-native computer vision system for aircraft surface defect detection. It runs on standard CPU hardware with zero internet dependency — from image capture through defect detection to PDF report generation.

## Target Defects

- Corrosion / Pitted Surface
- Paint Delamination / Patches
- Surface Cracks / Crazing / Scratches
- Fastener Anomalies / Inclusions / Rolled-in Scale

## Project Structure

```
├── data/                  # Datasets, annotations, splits
├── models/                # Trained weights, ONNX exports, model card
├── src/
│   ├── preprocessing/     # OpenCV image pipeline, CLAHE
│   ├── inference/         # ONNX Runtime wrapper, NMS, severity triage
│   ├── explainability/    # Grad-CAM on-demand module
│   ├── dashboard/         # Streamlit app
│   └── reporting/         # PDF report generator (fpdf2)
├── training/              # Training scripts, augmentation configs, logs
├── configs/               # Hyperparameters, model export settings
├── tests/                 # End-to-end and unit tests
├── docs/                  # Usage guide, model card, architecture notes
├── demo/                  # Demo assets, sample images, recording script
├── outputs/               # Generated reports, annotated images (gitignored)
├── requirements.txt       # Python dependencies
└── implementation_plan.md # Full build plan
```

## Quick Start

> _To be completed after Phase 4._

## License

> _To be determined._
