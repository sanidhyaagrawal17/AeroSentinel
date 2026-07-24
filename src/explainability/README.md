# /src/explainability

Grad-CAM visual attribution module.

## Responsibilities

- Load PyTorch `.pt` model on-demand (NOT the ONNX model)
- Hook into last conv layer of YOLOv8n backbone
- Generate Grad-CAM heatmap overlays on source images
- Unload model after use to free RAM

## Key Design Notes

- Grad-CAM requires PyTorch autograd graph — cannot run on ONNX models.
- This creates a dual-model architecture: ONNX for inference, PyTorch for explainability.
- On-demand loading is critical for 4GB RAM edge devices.
- Uses `pytorch-grad-cam` library.
