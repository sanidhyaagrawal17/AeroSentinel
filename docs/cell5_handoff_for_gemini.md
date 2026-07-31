# Handoff: Regenerate Cell 5 in `build_notebook.py`

## Task

Replace **only** Cell 5 (Static INT8 Quantization) in `d:\AeroSentinel TATA AIG 2026\docs\build_notebook.py`. Do not touch any other cell.

---

## File: `d:\AeroSentinel TATA AIG 2026\docs\build_notebook.py`

### What this file does

It's a Python script that programmatically builds a Jupyter notebook (`AeroSentinel_Training.ipynb`) by calling `md(...)` and `code(...)` helper functions. Each call appends a cell. The notebook has 7 logical cells (plus markdown headers between them).

### Lines to replace

**Cell 5 Header** — lines **535–537**:
```python
# =========================== CELL 5 HEADER ===========================
md("""---
## Cell 5: Static INT8 Quantization""")
```

**Cell 5 Code** — lines **539–646**:
```python
# =========================== CELL 5 ===========================
code("""# ============================================================
# Cell 5: Static INT8 Quantization
...
...
print(f'   Reduction:      {(1 - quant_size/orig_size)*100:.1f}%')""")
```

Replace lines **535–646** (inclusive) with the replacement block below.

---

## Replacement Block

> [!CAUTION]
> **Escaping rules** — the code lives inside a Python triple-quoted string (`code("""...""")`). You MUST:
> 1. Escape all interior triple-quotes as `\\\"\\\"\\\"` (the build script has a post-processing step on line 860 that converts `\\\"\\\"\\\"` back to `"""` in the final notebook JSON).
> 2. Escape literal `\n` in print statements as `\\\\n` (so the Python string contains `\\n`, which prints a newline at runtime).
> 3. Escape unicode escapes like `✅` as `\\\\u2705` (so the Python string contains `\u2705`).
> 4. Do NOT escape single quotes, single backslashes in comments, or anything else — only the three patterns above.

```python
# =========================== CELL 5 HEADER ===========================
md("""---
## Cell 5: Static INT8 Quantization (QDQ FIX APPLIED)
`quant_format` changed from `'QOperator'` to `'QDQ'`. The original format broke YOLO's detection head (DFL/concat/decode ops lack reliable quantized kernels), producing 0.0000 mAP across every class. QDQ inserts float fallback nodes for those ops. A sanity check now runs immediately after quantization to catch this failure mode before it reaches evaluation.""")

# =========================== CELL 5 ===========================
code("""# ============================================================
# Cell 5: Static INT8 Quantization
# ============================================================

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import onnx


class Preprocessor:
    \\\"\\\"\\\"
    EXACT replica of src/preprocessing/pipeline.py Preprocessor.
    CRITICAL: This must match the inference preprocessing pipeline exactly,
    otherwise calibration data will not represent runtime input distribution.
    \\\"\\\"\\\"
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.img_size = 640

    def process(self, image_source):
        if isinstance(image_source, (str, Path)):
            img0 = cv2.imread(str(image_source))
        else:
            img0 = image_source

        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        img, ratio, pad = self._letterbox(img_clahe, new_shape=(self.img_size, self.img_size))

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = img.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, img0, ratio, pad

    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)


class YoloCalibrationDataReader(CalibrationDataReader):
    \\\"\\\"\\\"Custom calibration reader that uses the same Preprocessor as inference.\\\"\\\"\\\"
    def __init__(self, image_folder, image_count, input_name):
        super().__init__()
        self.image_folder = Path(image_folder)
        self.input_name = input_name
        self.image_files = list(self.image_folder.glob('*.jpg')) + list(self.image_folder.glob('*.png'))
        self.image_files = self.image_files[:image_count]
        self.preprocessor = Preprocessor()
        self.enum_data = iter(self.image_files)

    def get_next(self):
        try:
            img_path = next(self.enum_data)
            tensor, _, _, _ = self.preprocessor.process(str(img_path))
            return {self.input_name: tensor}
        except StopIteration:
            return None


# ---- Paths ----
model_path = MODELS_DIR / 'aerosentinel.onnx'
output_path = MODELS_DIR / 'aerosentinel_int8.onnx'
calib_dir = PROCESSED_DIR / 'train' / 'images'
calib_count = 200

print(f'Loading ONNX model: {model_path}')
session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ---- Exclude the ENTIRE detection head (all /model.22/ nodes) ----
# Root cause (confirmed via intermediate-tensor debugging): quantizing any part of the
# classification branch (cv3.x convs, Concat_7, Sigmoid) zeroes confidence to exactly
# 0.0000, even when individual nodes are excluded one at a time -- QDQ boundary nodes
# kept reappearing at each new edge. Box regression (cv2.x / DFL) is unaffected either way.
# quant_pre_process() was tried as a prerequisite and FAILS with "Incomplete symbolic
# shape inference" (known ONNX Runtime limitation with YOLO's dynamic-shape DFL decode
# ops) -- skipped entirely here.
print('Building exclusion list for detection head (/model.22/)...')
model_check = onnx.load(str(model_path))
nodes_to_exclude = [node.name for node in model_check.graph.node if node.name.startswith('/model.22/')]
print(f'Excluding {len(nodes_to_exclude)} detection head nodes from quantization')

print(f'Calibrating with {calib_count} images from {calib_dir}...')
dr = YoloCalibrationDataReader(calib_dir, calib_count, input_name)

print('Starting quantize_static() -- this should take 1-3 minutes...')
quantize_static(
    model_input=str(model_path),
    model_output=str(output_path),
    calibration_data_reader=dr,
    quant_format='QDQ',
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    nodes_to_exclude=nodes_to_exclude,
)
print('quantize_static() finished.')

# ---- Sanity check: validate BOX outputs AND CONFIDENCE channels separately ----
# A generic "max > 0" check is NOT sufficient -- box coordinates survive quantization
# fine even when confidence is silently zeroed. Check both independently.
print('Running sanity check on quantized model...')
test_session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
test_tensor, _, _, _ = dr.preprocessor.process(str(dr.image_files[0]))
test_output = test_session.run(None, {input_name: test_tensor})[0]

box_max = test_output[0, :4, :].max()
conf_max = test_output[0, 4:, :].max()

print(f'  Box coordinate range: [{test_output[0, :4, :].min():.4f}, {box_max:.4f}]')
print(f'  Confidence channel range: [{test_output[0, 4:, :].min():.4f}, {conf_max:.4f}]')

assert box_max > 0, 'INT8 model produced all-zero box outputs -- quantization is broken'
assert conf_max > 0, 'INT8 model produced all-zero CONFIDENCE outputs -- classification head is broken'
print('\\\\u2705 Sanity check passed -- both box and confidence outputs are non-zero')

orig_size = os.path.getsize(model_path) / (1024 * 1024)
quant_size = os.path.getsize(output_path) / (1024 * 1024)

print(f'\\\\n\\\\u2705 Quantization complete.')
print(f'   Original size:  {orig_size:.2f} MB')
print(f'   Quantized size: {quant_size:.2f} MB')
print(f'   Reduction:      {(1 - quant_size/orig_size)*100:.1f}%')""")
```

---

## What Changed and Why

| # | Change | Reason |
|---|--------|--------|
| 1 | `quant_format='QOperator'` → `'QDQ'` | QOperator lacks float fallback kernels for YOLO's DFL/concat/decode ops — produces 0.0000 confidence on all detections. QDQ inserts QuantizeLinear/DequantizeLinear pairs that gracefully fall back to FP32 for unsupported ops. |
| 2 | Added `import onnx` | Needed to load the ONNX model graph and programmatically extract all `/model.22/` node names. |
| 3 | Added `nodes_to_exclude` — all 154 nodes under `/model.22/` | The detection head is precision-critical. Quantizing *any* part of the classification branch (cv3.x convs → Concat_7 → Sigmoid) zeroes confidence. Excluding individual nodes one-at-a-time failed because QDQ boundary nodes kept reappearing at each newly exposed edge. Excluding all 154 at once is the only approach that works. |
| 4 | Sanity check now validates box channels (`[0:4]`) AND confidence channels (`[4:]`) separately with individual `assert` statements | The original single `max() > 0` check was insufficient — box coordinates survive quantization fine even when confidence is silently zeroed to exactly 0.0000. This dual check is what caught the original bug. |
| 5 | Added progress prints before/after each major step | Enables pinpointing hangs to an exact line if quantization stalls (previous attempt hung 55+ minutes at 0% CPU/GPU). |
| 6 | Markdown header updated to document the fix | Explains QDQ rationale for anyone reading the notebook. |

## What's Preserved (Do NOT Change)

- `Preprocessor` class — **identical** to original, verified correct (CLAHE + letterbox matches inference pipeline)
- `YoloCalibrationDataReader` class — **identical**
- `calib_dir = PROCESSED_DIR / 'train' / 'images'`
- `calib_count = 200`
- `per_channel=True`, `weight_type=QuantType.QInt8`, `activation_type=QuantType.QUInt8`
- **All other cells** (1–4, 6–7) — untouched

## Approaches That Were Tried and Failed (Do NOT Re-attempt)

| Approach | Failure Mode |
|----------|-------------|
| `quant_pre_process()` before `quantize_static()` | `Exception: Incomplete symbolic shape inference` — known ONNX Runtime limitation with YOLO's dynamic-shape DFL decode ops (Slice/ConstantOfShape/Range). **Skip entirely.** |
| `quant_format='QOperator'` (original) | All confidence outputs = 0.0000. Box coords fine. DFL/concat/decode ops lack QOperator kernels. |
| Excluding individual `/model.22/` nodes one at a time | QDQ boundary nodes kept reappearing at each newly excluded node's edges. Concat before Sigmoid was the last leak found. Whack-a-mole. |
| `op_types_to_exclude` parameter | Not a supported parameter in `onnxruntime.quantization.quantize_static` as of v1.28.0. Only `nodes_to_exclude` and `nodes_to_quantize` exist. |
| Re-exporting with `dynamic=False` for static shape | Not attempted — the QDQ + full head exclusion approach worked first, making this unnecessary. |

## Confirmed Output (Kaggle T4, onnxruntime 1.28.0)

```
Loading ONNX model: /kaggle/working/aerosentinel/models/aerosentinel.onnx
Building exclusion list for detection head (/model.22/)...
Excluding 154 detection head nodes from quantization
Calibrating with 200 images from .../processed/train/images...
Starting quantize_static() — this should take 1-3 minutes...
quantize_static() finished.
Running sanity check on quantized model...
  Box coordinate range: [7.6981, 631.6640]
  Confidence channel range: [0.0000, 0.1393]
✅ Sanity check passed — both box and confidence outputs are non-zero

✅ Quantization complete.
   Original size:  11.87 MB
   Quantized size: 5.62 MB
   Reduction:      52.6%
```

## Post-Edit Validation Checklist

1. **Run `python docs/build_notebook.py`** — must complete without errors
2. **Open `training/AeroSentinel_Training.ipynb`** — must be valid JSON (open in Jupyter or `python -c "import json; json.load(open('training/AeroSentinel_Training.ipynb'))"`)
3. **Verify Cell 5 source** — check that `\\\"\\\"\\\"` was correctly converted to `"""` by the post-processing on line 860
4. **Verify no other cells were changed** — diff the notebook before/after, only the Cell 5 header markdown and Cell 5 code cell should differ

## Source of Truth

The working code was extracted from the verified Kaggle notebook:
`d:\AeroSentinel TATA AIG 2026\training\notebooka2ae66b251.ipynb` — Cell index 10 (code), with markdown header at Cell index 9.
