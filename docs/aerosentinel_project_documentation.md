# AeroSentinel: Steel Surface Defect Detection

## 1. Project Overview
**AeroSentinel** is an advanced computer vision project aimed at developing a highly accurate and lightweight defect detection model for industrial steel surfaces. The core objective is to build a state-of-the-art (SOTA) YOLOv8-based architecture capable of identifying microscopic and structural anomalies, with the eventual goal of deploying the model on edge devices (e.g., UAVs/drones) for automated industrial inspection.

---

## 2. Dataset Details: NEU-DET
The project relies on the **NEU-DET** (Northeastern University Surface Defect Database).

- **Resolution:** Native `200x200` pixels (Grayscale).
- **Total Images:** 1,800 (Split: 1259 Train, 360 Validation, 181 Test).
- **Classes (6):**
  1. `Crazing` (Irregular, fine cracks)
  2. `Inclusion` (Embedded metallic/non-metallic particles)
  3. `Patches` (Localized irregular textures)
  4. `Pitted Surface` (Small, crater-like depressions)
  5. `Rolled-in Scale` (Oxide scale pressed into the metal)
  6. `Scratches` (Linear surface abrasions)

> [!WARNING]
> **Dataset Challenges:** The dataset is extremely small (1,259 training images). Deep learning models like YOLOv8 are highly prone to **overfitting** on this dataset unless aggressive and mathematically optimal data augmentations are applied.

---

## 3. Literature Benchmark
A comprehensive review of published academic literature regarding YOLOv8 on the NEU-DET dataset reveals the following benchmarks:
- **Average SOTA Performance:** `77.0%` to `81.0%` mAP50.
- **Common Literature Enhancements:** Adding Attention mechanisms (CBAM, ECA), changing convolutions to Receptive-Field Attention Convolutions (RFAConv), and modifying the SPPF layer.

**Our Goal:** To match or exceed the `80%` mAP50 threshold before applying our own architectural enhancements.

---

## 4. Experimental History & Milestones

### Phase 1: The False Baseline (Offline Augmentation)
- **Action:** We initially generated a 5x expanded dataset using a custom offline Python pipeline that injected procedural textures, domain randomization, and synthetic distractors (blur, noise).
- **Result:** The model achieved `51.1%` mAP50.
- **Status:** *Discarded*.

### Phase 2: The Audit (Raw vs. Augmented)
- **Action:** Suspecting the offline augmentations were introducing destructive noise, we ran a strict head-to-head A/B test.
  - **Arm A (Raw Data):** Achieved **`55.0%`** mAP50.
  - **Arm B (Augmented Data):** Achieved **`47.8%`** mAP50.
- **Conclusion:** The synthetic offline augmentations were actively harming the model's ability to learn real-world textures. We permanently dropped the offline augmentation pipeline and pivoted to using purely raw data.

### Phase 3: The Bottleneck Hunt
- **Analysis:** Despite using raw data, our `55.0%` baseline was still far below the literature SOTA of `77%`.
- **Discovery:** We identified two massive bottlenecks in our training configuration:
  1. We were forcing `imgsz=640`, forcing YOLO to aggressively upscale the native 200x200 images, destroying high-frequency edge data.
  2. We were explicitly disabling YOLOv8's most powerful native augmentations (Mosaic, Mixup) and forcing custom optimizers (`AdamW` with constrained learning rates), completely preventing the model from generalizing.

### Phase 4: The 3-Way Ablation Study (CURRENT)
To establish the ultimate unbreakable baseline, we are currently running a 3-arm automated hyperparameter experiment on Lightning AI:
- **Arm C (`imgsz=640` + Native Defaults):** *Completed.* Achieved **`55.3%`** mAP50. Proved that defaults alone cannot overcome the upscaling bottleneck.
- **Arm D (`imgsz=320` + Old Constraints):** *Running.* Testing if native resolution alone fixes the issue.
- **Arm E (`imgsz=320` + Native Defaults):** *Pending.* The combined test. Expected to yield the highest baseline score.

---

## 5. Future Roadmap

Once the 3-Way Ablation Study finishes and we have secured our optimal baseline configuration (aiming for `~75-80%` mAP50), the project will advance to the following phases:

### A. Architectural Modifications
To push beyond the literature SOTA, we will modify the underlying YOLOv8 architecture:
- **RFAConv (Receptive-Field Attention Convolutions):** Replacing standard Conv layers in the YOLOv8 backbone to better capture the microscopic, irregular nature of defects like *Crazing*.
- **Attention Modules:** Integrating `CBAM` or `BiFormer` attention gates in the neck to help the model focus on faint scratches against noisy steel backgrounds.

### B. Deployment & Edge Optimization
- **Exporting:** Converting the final PyTorch (`.pt`) model to ONNX and TensorRT formats.
- **Quantization:** Applying INT8/FP16 quantization to drastically reduce the model's memory footprint while maintaining >95% of its mAP score.
- **Inference:** Ensuring the model can run at >30 FPS on edge-compute hardware (e.g., NVIDIA Jetson) for real-time drone-based surveillance.
