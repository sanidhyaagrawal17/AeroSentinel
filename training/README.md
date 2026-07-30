# /training

Training scripts, augmentation configs, and training logs.

## Expected Contents

| File | Description |
|------|-------------|
| `train.py` | YOLOv8n fine-tuning script (Ultralytics CLI or API) |
| `augmentation.py` | Albumentations pipeline definition |
| `export_onnx.py` | PyTorch → ONNX export script |
| `quantize.py` | ONNX INT8 quantization script |
| `evaluate.py` | mAP evaluation + per-class AP breakdown |
| `runs/` | Ultralytics training run outputs (auto-generated) |

## Training Specs
- **Base model:** YOLOv8n (COCO-pretrained)
- **Epochs:** 100–150
- **Batch size:** 16
- **Image size:** 640
- **Optimizer:** SGD
- **Target:** ≥85% mAP@0.5
## Augmentations (Albumentations)
- Random brightness/contrast
- Perspective warp
- CLAHE
- Horizontal flip
- Synthetic crack overlay (stretch goal)
