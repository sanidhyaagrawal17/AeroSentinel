# /configs

Centralized configuration files for all pipeline components.

## Expected Contents

| File | Description |
|------|-------------|
| `train_config.yaml` | Hyperparameters: epochs, batch, lr, img_size, optimizer |
| `inference_config.yaml` | Confidence threshold, NMS IoU threshold, severity rules |
| `export_config.yaml` | ONNX export settings: dynamic axes, simplify, opset |
| `quantize_config.yaml` | INT8 calibration: num_calibration_images, quant_format |
| `clahe_config.yaml` | Clip limit, tile grid size |
