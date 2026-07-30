# Deprecated Experiments

This directory contains code and configurations from experimental approaches that were tested during the development of AeroSentinel but ultimately abandoned. They are preserved here as a record of our engineering rigor and experimentation process.

## 1. RFAConv Attention Module (`RFAConv_model.py`, `yolov8-rfaconv.yaml`, `run_rfa_experiment.py`)
We attempted to inject a custom Receptive-Field Attention Convolution (RFAConv) module into the YOLOv8 architecture to improve spatial context for complex defect textures. 
**Result:** Abandoned. The custom module significantly increased training time and complexity without yielding any measurable accuracy benefit (mAP50 remained stagnant compared to the baseline).

## 2. Offline Synthetic Augmentation (`generate_synthetic.py`, `augmentation.py`)
We built a pipeline to synthesize defects using advanced blending techniques (Poisson blending, Seamless Cloning, Gaussian noise) to artificially expand the dataset.
**Result:** Abandoned. Controlled 300-epoch testing conclusively proved this approach actively harmed accuracy. The model overfitted to the synthetic blending artifacts rather than the actual defect textures (mAP50 dropped from 55.0% on raw data to 47.8% on augmented data).

## 3. Data Generation Sandbox (`test_generation.py`, `run_experiments.py`)
Sandbox scripts used early on to test various generation and evaluation strategies.
**Result:** Deprecated in favor of the unified pipeline and hyperparameter tuning (GA) approach which yielded our current 58.1% mAP50 baseline.

## 4. Old Training Notebook (`build_notebook.py`, `AeroSentinel_Training.ipynb`)
The original Kaggle training notebook which implemented the abandoned 5x offline augmentation pipeline and earlier training configs. 
**Result:** Deprecated. It is preserved here for historical context on how the 51.1% pipeline was trained, but it does NOT reflect our final GA-tuned, raw-data configuration that achieved the 58.1% baseline.
