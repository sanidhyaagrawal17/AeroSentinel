# /models

Stores all model artifacts across the training → export → quantization pipeline.

## Expected Contents

| File | Phase | Description |
|------|-------|-------------|
| `best.pt` | Phase 1 | Best YOLOv8n PyTorch checkpoint (also used for Grad-CAM) |
| `last.pt` | Phase 1 | Final epoch checkpoint |
| `aerosentinel.onnx` | Phase 2 | ONNX export (FP32, dynamic axes) |
| `aerosentinel_int8.onnx` | Phase 2 | INT8 quantized ONNX (<2MB target) |
| `model_card.md` | Phase 5 | Model card (see template in implementation plan) |

## Notes

- The `.pt` file is retained for on-demand Grad-CAM explainability only.
- Primary inference uses the INT8 ONNX model via ONNX Runtime.
- All files here are gitignored except `model_card.md`.
