# /src/inference

ONNX Runtime inference engine with post-processing.

## Responsibilities

- Load INT8 ONNX model via `onnxruntime.InferenceSession`
- Run single-pass detection on preprocessed frames
- Non-Maximum Suppression (NMS) post-processing
- Confidence threshold filtering
- Severity triage: rule-based heuristic (area × confidence → Low / Medium / Critical)

## Key Design Notes

- Target: <30ms per frame on i5-class CPU, CPU ExecutionProvider only.
- Severity triage is **not** a learned classifier — documented as a PoC-scope heuristic.
- NMS IoU threshold and confidence threshold should be configurable via `configs/`.
