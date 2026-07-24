# AeroSentinel Model Card

| Field | Content |
|-------|---------|
| **Model** | YOLOv8n (Nano) fine-tuned |
| **Format** | ONNX, INT8 quantized |
| **Training Data** | NEU Metal Surface Defect Dataset (1,800 images, 6 classes) |
| **Defect Classes** | Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches |
| **Performance** | ≥85% mAP@0.5 on test set (pre-quantization); ≤2% drop post-INT8 |
| **Intended Use** | PoC demonstration of edge-native aircraft surface inspection |
| **Out-of-Scope Uses** | Production MRO certification; real aerospace structural sign-off |
| **Known Limitations** | Trained on flat steel surfaces — domain gap to curved aerospace surfaces. Severity triage is heuristic, not a learned classifier. |
| **Hardware Requirements** | Any x86 CPU (i5-class or above); no GPU required; 4GB+ RAM |

## Class Mapping (NEU to Aerospace MRO)
- Crazing -> Cracks
- Inclusion -> Fastener Anomaly
- Patches -> Delamination
- Pitted Surface -> Corrosion
- Rolled-in Scale -> Fastener Anomaly
- Scratches -> Cracks
