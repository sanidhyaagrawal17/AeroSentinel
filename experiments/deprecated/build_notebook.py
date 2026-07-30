"""
Generate the updated AeroSentinel_Training.ipynb with accuracy improvements.
"""
import json

cells = []

def _split_source(source):
    """Split source into lines with \\n appended (Jupyter nbformat spec)."""
    lines = source.split('\n')
    # Every line except the last gets a trailing \n
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _split_source(source)})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": _split_source(source), "outputs": [], "execution_count": None})

# =========================== HEADER ===========================
md("""# \U0001f6e1\ufe0f AeroSentinel \u2014 Full Training & Optimization Pipeline
**Team Cygnus | TATA AIG 2026**

This notebook runs the complete pipeline on a **Kaggle T4 GPU**:
1. Download NEU-DET dataset
2. Apply augmentations (5x data multiplication + synthetic cracks)
3. Train **YOLOv8n** (Nano) with optimized hyperparameters
4. Export to ONNX
5. INT8 quantization
6. Evaluate mAP (FP32 vs INT8)
7. Save artifacts to Kaggle output

**Key Accuracy Improvements (v2):**
- \u2705 YOLOv8n (Nano, 3.2M params) \u2014 smallest footprint, ~3.4MB INT8
- \u2705 5x offline augmentation (data multiplication)
- \u2705 AdamW optimizer with lower LR (0.001)
- \u2705 Backbone freezing for first 10 epochs
- \u2705 Mosaic + MixUp built-in augmentations
- \u2705 300 epochs with patience=75
- \u2705 Class-aware synthetic crack overlay (only on Crazing/Scratches)

> **Kaggle Note:** Artifacts are saved to `/kaggle/working/`. After the run completes, download them from the **Output** tab.""")

# =========================== CELL 1 HEADER ===========================
md("""---
## Cell 1: Environment Setup & Dataset Download""")

# =========================== CELL 1 ===========================
code("""# ============================================================
# Cell 1: Environment Setup & Dataset Download
# ============================================================

!pip install -q ultralytics onnx onnxruntime onnxruntime-gpu albumentations roboflow fpdf2 opencv-python-headless

import os
import shutil
from pathlib import Path
from roboflow import Roboflow

# --- Configuration ---
PROJECT_ROOT = Path('/kaggle/working/aerosentinel')
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

for d in [PROJECT_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Download NEU-DET from Roboflow ---
API_KEY = os.getenv('ROBOFLOW_API_KEY', 'your-api-key-here')

print('Authenticating with Roboflow...')
rf = Roboflow(api_key=API_KEY)
project = rf.workspace('park-sung-ho').project('neu-det-object-detection')
version = project.version(1)

print('Downloading YOLOv8 dataset...')
dataset = version.download('yolov8')
roboflow_dir = Path(dataset.location)

print(f'Moving dataset from {roboflow_dir} to {RAW_DIR}...')
for split in ['train', 'valid', 'test']:
    src = roboflow_dir / split
    dst = RAW_DIR / split
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

try:
    shutil.rmtree(roboflow_dir)
except:
    pass

# --- Verify dataset structure ---
for split in ['train', 'valid', 'test']:
    img_dir = RAW_DIR / split / 'images'
    lbl_dir = RAW_DIR / split / 'labels'
    if img_dir.exists():
        img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        lbl_count = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
        print(f'  {split}: {img_count} images, {lbl_count} labels')
    else:
        print(f'  {split}: NOT FOUND')

print('\\n\\u2705 Dataset download complete.')""")

# =========================== CELL 2 HEADER ===========================
md("""---
## Cell 2: 5x Data Multiplication + Class-Aware Augmentation

**Key changes from v1:**
- Generates **5 augmented copies** per training image (up from 1x)
- Synthetic crack overlay **only applied to Crazing & Scratches classes** (not all classes)
- Validation/test images kept pristine (no augmentation)""")

# =========================== CELL 2 ===========================
code("""# ============================================================
# Cell 2: 5x Data Multiplication + Class-Aware Augmentation
# ============================================================

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import math
import random
import matplotlib.pyplot as plt

# ---- Synthetic Crack Overlay Transform ----
class SyntheticCrackOverlay(ImageOnlyTransform):
    \\"\\"\\"
    Custom Albumentations transform that draws realistic synthetic crack
    patterns onto training images. Uses a random-walk polyline algorithm
    with branching to simulate surface cracks on metallic surfaces.
    \\"\\"\\"
    def __init__(self, num_cracks=(1, 3), num_steps=(20, 60), step_length=(3, 8),
                 thickness=(1, 3), angle_perturbation=15, branch_prob=0.3,
                 color_range=(30, 80), blur_kernel=3, always_apply=False, p=0.4):
        super().__init__(always_apply=always_apply, p=p)
        self.num_cracks = num_cracks
        self.num_steps = num_steps
        self.step_length = step_length
        self.thickness = thickness
        self.angle_perturbation = angle_perturbation
        self.branch_prob = branch_prob
        self.color_range = color_range
        self.blur_kernel = blur_kernel

    def _draw_crack_path(self, overlay, h, w):
        \\"\\"\\"Draw a single crack path using random walk.\\"\\"\\"
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        angle = random.uniform(0, 2 * math.pi)

        steps = random.randint(*self.num_steps)
        thickness = random.randint(*self.thickness)
        color_val = random.randint(*self.color_range)
        color = (color_val, color_val, color_val)

        points = [(x, y)]
        branch_points = []

        for step_i in range(steps):
            angle += math.radians(random.uniform(-self.angle_perturbation, self.angle_perturbation))
            step_len = random.uniform(*self.step_length)

            x_new = int(x + step_len * math.cos(angle))
            y_new = int(y + step_len * math.sin(angle))
            x_new = max(0, min(w - 1, x_new))
            y_new = max(0, min(h - 1, y_new))

            cv2.line(overlay, (x, y), (x_new, y_new), color, thickness)
            points.append((x_new, y_new))

            if step_i > 5 and random.random() < 0.1:
                branch_points.append((x_new, y_new, angle))

            x, y = x_new, y_new

        # Draw 1-2 branches if probability allows
        if random.random() < self.branch_prob and branch_points:
            num_branches = min(random.randint(1, 2), len(branch_points))
            for bp in random.sample(branch_points, num_branches):
                bx, by, ba = bp
                ba += math.radians(random.uniform(-45, 45))
                branch_steps = random.randint(8, 20)
                branch_thickness = max(1, thickness - 1)
                for _ in range(branch_steps):
                    ba += math.radians(random.uniform(-self.angle_perturbation, self.angle_perturbation))
                    sl = random.uniform(*self.step_length) * 0.7
                    bx_new = int(bx + sl * math.cos(ba))
                    by_new = int(by + sl * math.sin(ba))
                    bx_new = max(0, min(w - 1, bx_new))
                    by_new = max(0, min(h - 1, by_new))
                    cv2.line(overlay, (bx, by), (bx_new, by_new), color, branch_thickness)
                    bx, by = bx_new, by_new

    def apply(self, img, **params):
        h, w = img.shape[:2]
        overlay = img.copy()

        num = random.randint(*self.num_cracks)
        for _ in range(num):
            self._draw_crack_path(overlay, h, w)

        if self.blur_kernel > 0:
            k = self.blur_kernel
            overlay = cv2.GaussianBlur(overlay, (k, k), 0)
            mask = cv2.absdiff(img, overlay)
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
            _, mask_bin = cv2.threshold(mask_gray, 5, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.merge([mask_bin, mask_bin, mask_bin]) if len(img.shape) == 3 else mask_bin
            result = np.where(mask_3ch > 0, overlay, img)
            return result.astype(np.uint8)

        return overlay

    def get_transform_init_args_names(self):
        return ('num_cracks', 'num_steps', 'step_length', 'thickness',
                'angle_perturbation', 'branch_prob', 'color_range', 'blur_kernel')


# ---- Classes that should get synthetic crack overlay (Crazing=0, Scratches=5) ----
CRACK_CLASS_IDS = {0, 5}

# ---- Base augmentation pipeline (applied to ALL training images) ----
base_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.GaussNoise(p=1.0),
    ], p=0.3),
])

# ---- Crack-class augmentation (adds synthetic cracks on top of base) ----
crack_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    SyntheticCrackOverlay(p=0.5),  # Only for crack classes
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.GaussNoise(p=1.0),
    ], p=0.3),
])


def get_class_ids_from_label(label_path):
    \\"\\"\\"Read YOLO label file and return set of class IDs present.\\"\\"\\"
    class_ids = set()
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return class_ids


# ---- Number of augmented copies per training image ----
NUM_COPIES = 5

# ---- Apply Offline Augmentation ----
count = 0
for split in ['train', 'valid', 'test']:
    raw_img_dir = RAW_DIR / split / 'images'
    raw_lbl_dir = RAW_DIR / split / 'labels'
    proc_img_dir = PROCESSED_DIR / split / 'images'
    proc_lbl_dir = PROCESSED_DIR / split / 'labels'

    if not raw_img_dir.exists():
        print(f'  Skipping {split} (not found)')
        continue

    proc_img_dir.mkdir(parents=True, exist_ok=True)
    proc_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(raw_img_dir.glob('*.jpg')) + list(raw_img_dir.glob('*.png')))
    print(f'Processing split: {split} ({len(img_files)} images)...')

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = raw_lbl_dir / f'{img_path.stem}.txt'

        if split == 'train':
            # Determine which augmentation pipeline to use
            class_ids = get_class_ids_from_label(label_path)
            is_crack_class = bool(class_ids & CRACK_CLASS_IDS)
            aug_pipeline = crack_aug if is_crack_class else base_aug

            # Generate NUM_COPIES augmented versions
            for copy_idx in range(NUM_COPIES):
                augmented = aug_pipeline(image=img)
                aug_img = augmented['image']

                # Save with copy index suffix
                out_name = f'{img_path.stem}_aug{copy_idx}{img_path.suffix}'
                if img_path.suffix.lower() == '.png':
                    out_name = f'{img_path.stem}_aug{copy_idx}.jpg'

                cv2.imwrite(str(proc_img_dir / out_name), aug_img)

                # Copy label with matching name
                if label_path.exists():
                    lbl_out_name = f'{img_path.stem}_aug{copy_idx}.txt'
                    shutil.copy2(label_path, proc_lbl_dir / lbl_out_name)

                count += 1

            # Also keep the original (unaugmented) for diversity
            cv2.imwrite(str(proc_img_dir / f'{img_path.stem}_orig.jpg'), img)
            if label_path.exists():
                shutil.copy2(label_path, proc_lbl_dir / f'{img_path.stem}_orig.txt')
            count += 1

        else:
            # val/test: keep pristine (no augmentation)
            out_name = f'{img_path.stem}.jpg'
            cv2.imwrite(str(proc_img_dir / out_name), img)
            if label_path.exists():
                shutil.copy2(label_path, proc_lbl_dir / f'{img_path.stem}.txt')
            count += 1

# Print final stats
for split in ['train', 'valid', 'test']:
    d = PROCESSED_DIR / split / 'images'
    if d.exists():
        n = len(list(d.glob('*.jpg')))
        print(f'  {split}: {n} images')

print(f'\\n\\u2705 Offline augmentation complete. Total processed: {count} images.')
print(f'   Training data multiplied {NUM_COPIES + 1}x (5 augmented + 1 original per source image)')


# ---- Visual Verification: Show before/after with synthetic cracks ----
print('\\n--- Visual Verification: Synthetic Crack Augmentation ---')
train_imgs = sorted((RAW_DIR / 'train' / 'images').glob('*.jpg'))[:4]

crack_only = SyntheticCrackOverlay(p=1.0, always_apply=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, img_path in enumerate(train_imgs):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cracked = crack_only(image=img)['image']
    cracked_rgb = cv2.cvtColor(cracked, cv2.COLOR_BGR2RGB)

    axes[0, i].imshow(img_rgb)
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')
    axes[1, i].imshow(cracked_rgb)
    axes[1, i].set_title(f'+ Synthetic Cracks {i+1}')
    axes[1, i].axis('off')

plt.suptitle('Synthetic Crack Augmentation \\u2014 Before vs After', fontsize=14)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'synthetic_cracks_preview.png'), dpi=150)
plt.show()
print('\\u2705 Preview saved.')""")

# =========================== CELL 3 HEADER ===========================
md("""---
## Cell 3: Train YOLOv8n on Kaggle T4 x2

**Key changes from v1:**
- **YOLOv8n** (Nano, 3.2M params) \u2014 smallest YOLO for ~3.4MB INT8 footprint
- **AdamW** optimizer (lr=0.001) instead of SGD (lr=0.01) \u2014 converges better on small datasets
- **300 epochs** with patience=75 (up from 100/50)
- **Backbone freeze** for first 5 epochs to preserve COCO feature representations
- **YOLO built-in heavy augmentations disabled** (mosaic/mixup/copy_paste/degrees=0) \u2014 offline 5x augmentation already applied
- **Image size 640**""")

# =========================== CELL 3 ===========================
code("""# ============================================================
# Cell 3: Train YOLOv8n on T4 GPU (Optimized Hyperparameters)
# ============================================================

import yaml
from ultralytics import YOLO

# ---- Create data.yaml pointing to processed directories ----
data_yaml_content = {
    'path': str(PROCESSED_DIR),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': {
        0: 'Crazing',
        1: 'Inclusion',
        2: 'Patches',
        3: 'Pitted_Surface',
        4: 'Rolled_in_Scale',
        5: 'Scratches'
    }
}

data_yaml_path = DATA_DIR / 'data.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f'data.yaml written to {data_yaml_path}')

# ---- Load pre-trained YOLOv8n and fine-tune ----
# YOLOv8n (Nano, 3.2M params) gives ~3.4MB INT8 footprint
# All accuracy gains come from better training strategy, not larger model
print('Loading YOLOv8n pre-trained model...')
model = YOLO('yolov8n.pt')

# ---- Optimized Training Configuration ----
print('Starting training on Kaggle T4 GPU with optimized hyperparameters...')
results = model.train(
    data=str(data_yaml_path),

    # --- Core ---
    epochs=300,           # More time to converge on small dataset; early stopping handles the rest
    batch=16,
    imgsz=640,
    device=[0,1],         # T4 x2 GPUs on Kaggle

    # --- Optimizer (AdamW instead of SGD) ---
    optimizer='AdamW',    # Better convergence on small datasets vs SGD
    lr0=0.001,            # Lower LR for fine-tuning (was 0.01)
    lrf=0.01,             # Final LR = lr0 * lrf = 0.00001
    weight_decay=0.0005,
    warmup_epochs=5,      # Gradual LR warmup to avoid shocking pre-trained weights

    # --- Early Stopping ---
    patience=75,          # Up from 50: small dataset loss can be noisy

    # --- Transfer Learning ---
    freeze=5,             # Freeze backbone for first 5 epochs
                          # Lets detection head adapt before full fine-tuning

    # --- Built-in Augmentations ---
    # DISABLED heavy augmentations: offline 5x augmentation already applied in Cell 2.
    # Stacking mosaic/mixup on top of already-augmented images causes double-distortion.
    mosaic=0,             # DISABLED: was 1.0 — caused double augmentation
    mixup=0,              # DISABLED: was 0.15 — blending blurs small defects
    copy_paste=0,         # DISABLED: was 0.1
    degrees=0,            # DISABLED: was 10.0 — rotation on already-warped images
    scale=0.5,            # Random scale (kept)
    fliplr=0.5,           # Horizontal flip (kept)
    flipud=0,             # DISABLED: was 0.3 — vertical flip compounds distortion
    hsv_h=0.015,          # Hue augmentation (kept)
    hsv_s=0.7,            # Saturation augmentation (kept)
    hsv_v=0.4,            # Value/brightness augmentation (kept)

    # --- Output ---
    project=str(PROJECT_ROOT / 'runs'),
    name='aerosentinel_train_v2',
    plots=True,           # Generate training plots
    save=True,
    exist_ok=True,
)

# ---- FIX: Handle DDP multi-GPU return type ----
# When training with device=[0,1], Ultralytics returns a dict instead of a Results object.
# results.save_dir crashes with AttributeError on a dict. Fall back to the known path.
if hasattr(results, 'save_dir'):
    save_dir = Path(results.save_dir)
else:
    # Fallback: use the project/name path we specified in model.train()
    save_dir = PROJECT_ROOT / 'runs' / 'aerosentinel_train_v2'
    print(f'Note: DDP returned dict, using fallback save_dir: {save_dir}')

print(f'\\n\\u2705 Training complete!')
print(f'Best model: {save_dir}/weights/best.pt')

# Copy best.pt to models/
best_pt_path = save_dir / 'weights' / 'best.pt'
target_pt_path = MODELS_DIR / 'best.pt'
if best_pt_path.exists():
    shutil.copy2(best_pt_path, target_pt_path)
    print(f'Copied best.pt to {target_pt_path}')
    model_size = os.path.getsize(target_pt_path) / (1024 * 1024)
    print(f'Model size: {model_size:.2f} MB')
else:
    print(f'ERROR: best.pt not found at {best_pt_path}')

# Also copy training plots
for plot_file in save_dir.glob('*.png'):
    shutil.copy2(plot_file, PROJECT_ROOT / plot_file.name)
    print(f'Copied plot: {plot_file.name}')""")

# =========================== CELL 4 HEADER ===========================
md("""---
## Cell 4: Export to ONNX""")

# =========================== CELL 4 ===========================
code("""# ============================================================
# Cell 4: Export to ONNX
# ============================================================

from ultralytics import YOLO

print('Loading trained model for ONNX export...')
model = YOLO(str(MODELS_DIR / 'best.pt'))

print('Exporting to ONNX (opset 13, dynamic axes, simplified)...')
export_path = model.export(
    format='onnx',
    dynamic=True,
    simplify=True,
    imgsz=640,
    opset=13,
    half=False
)

# Move to models directory
onnx_target = MODELS_DIR / 'aerosentinel.onnx'
if Path(export_path).exists():
    shutil.copy2(export_path, onnx_target)
    os.unlink(export_path)
    onnx_size_mb = os.path.getsize(onnx_target) / (1024 * 1024)
    print(f'\\n\\u2705 ONNX model saved to {onnx_target}')
    print(f'   Size: {onnx_size_mb:.2f} MB')
else:
    print(f'ERROR: Export failed, {export_path} not found')""")

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
    \\"\\"\\"
    EXACT replica of src/preprocessing/pipeline.py Preprocessor.
    CRITICAL: This must match the inference preprocessing pipeline exactly,
    otherwise calibration data will not represent runtime input distribution.
    \\"\\"\\"
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
    \\"\\"\\"Custom calibration reader that uses the same Preprocessor as inference.\\"\\"\\"
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

# =========================== CELL 6 HEADER ===========================
md("""---
## Cell 6: Evaluate mAP (FP32 + INT8)""")

# =========================== CELL 6 ===========================
code("""# ============================================================
# Cell 6: Evaluate mAP (FP32 + INT8)
# ============================================================

from ultralytics import YOLO
import datetime

data_yaml_path = DATA_DIR / 'data.yaml'

# ---- 6a: FP32 baseline (PyTorch) ----
print('=' * 60)
print('Evaluating FP32 model (best.pt)...')
print('=' * 60)
fp32_model = YOLO(str(MODELS_DIR / 'best.pt'))
fp32_metrics = fp32_model.val(data=str(data_yaml_path))

fp32_map50 = fp32_metrics.box.map50
fp32_map50_95 = fp32_metrics.box.map
fp32_map75 = fp32_metrics.box.map75

print(f'\\nFP32 Results:')
print(f'  mAP@0.5:      {fp32_map50:.4f}')
print(f'  mAP@0.5:0.95: {fp32_map50_95:.4f}')
print(f'  mAP@0.75:     {fp32_map75:.4f}')

# Per-class AP
class_names = list(fp32_metrics.names.values())
fp32_per_class = {}
for i, name in enumerate(class_names):
    try:
        fp32_per_class[name] = fp32_metrics.box.maps[i]
        print(f'  {name}: {fp32_metrics.box.maps[i]:.4f}')
    except IndexError:
        fp32_per_class[name] = 'N/A'

# ---- 6b: INT8 validation ----
print('\\n' + '=' * 60)
print('Evaluating INT8 model (aerosentinel_int8.onnx)...')
print('=' * 60)
int8_model = YOLO(str(MODELS_DIR / 'aerosentinel_int8.onnx'))
int8_metrics = int8_model.val(data=str(data_yaml_path))

int8_map50 = int8_metrics.box.map50
int8_map50_95 = int8_metrics.box.map
int8_map75 = int8_metrics.box.map75

print(f'\\nINT8 Results:')
print(f'  mAP@0.5:      {int8_map50:.4f}')
print(f'  mAP@0.5:0.95: {int8_map50_95:.4f}')
print(f'  mAP@0.75:     {int8_map75:.4f}')

int8_per_class = {}
for i, name in enumerate(class_names):
    try:
        int8_per_class[name] = int8_metrics.box.maps[i]
        print(f'  {name}: {int8_metrics.box.maps[i]:.4f}')
    except IndexError:
        int8_per_class[name] = 'N/A'

# ---- Compute drops ----
drop_map50 = fp32_map50 - int8_map50
drop_map50_pct = (drop_map50 / fp32_map50) * 100 if fp32_map50 > 0 else 0
drop_map50_95 = fp32_map50_95 - int8_map50_95

# ---- Generate evaluation_results.md ----
fp32_size_mb = os.path.getsize(MODELS_DIR / 'best.pt') / (1024 * 1024)
int8_size_mb = os.path.getsize(MODELS_DIR / 'aerosentinel_int8.onnx') / (1024 * 1024)

report_lines = [
    '# AeroSentinel \\u2014 Evaluation Results',
    f'*Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
    '',
    '## Overall Metrics',
    '',
    '| Metric | FP32 (best.pt) | INT8 (aerosentinel_int8.onnx) | Drop |',
    '|--------|----------------|-------------------------------|------|',
    f'| mAP@0.5 | {fp32_map50:.4f} | {int8_map50:.4f} | {drop_map50:.4f} ({drop_map50_pct:.1f}%) |',
    f'| mAP@0.5:0.95 | {fp32_map50_95:.4f} | {int8_map50_95:.4f} | {drop_map50_95:.4f} |',
    f'| mAP@0.75 | {fp32_map75:.4f} | {int8_map75:.4f} | {fp32_map75 - int8_map75:.4f} |',
    f'| Model Size | {fp32_size_mb:.2f} MB | {int8_size_mb:.2f} MB | -{(1 - int8_size_mb/fp32_size_mb)*100:.1f}% |',
    '',
    '## Per-Class AP@0.5:0.95',
    '',
    '| Class | FP32 | INT8 | Drop |',
    '|-------|------|------|------|',
]

for name in class_names:
    fp32_val = fp32_per_class.get(name, 'N/A')
    int8_val = int8_per_class.get(name, 'N/A')
    if isinstance(fp32_val, float) and isinstance(int8_val, float):
        drop = fp32_val - int8_val
        report_lines.append(f'| {name} | {fp32_val:.4f} | {int8_val:.4f} | {drop:.4f} |')
    else:
        report_lines.append(f'| {name} | {fp32_val} | {int8_val} | N/A |')

report_lines.extend([
    '',
    '## Configuration',
    '',
    '| Parameter | Value |',
    '|-----------|-------|',
    '| Architecture | YOLOv8n (Nano) |',
    '| Training Epochs | 300 (with early stopping, patience=75) |',
    '| Training Device | Kaggle T4 GPU |',
    '| Optimizer | AdamW (lr=0.001) |',
    '| Backbone Freeze | 10 epochs |',
    '| Data Augmentation | 5x offline + mosaic + mixup |',
    '| Quantization | Static (INT8, per-channel, QOperator) |',
    '| Calibration Images | 200 |',
    '| ONNX Opset | 13 |',
    f'| Training Dataset | NEU-DET (6 classes, 5x augmented) |',
])

eval_report_path = MODELS_DIR / 'evaluation_results.md'
with open(eval_report_path, 'w') as f:
    f.write('\\n'.join(report_lines))

print(f'\\n\\u2705 Evaluation report saved to {eval_report_path}')
print(f'\\n--- KEY CLAIMS VALIDATION ---')
print(f'  mAP@0.5 >= 85%?   {"\\u2705 YES" if fp32_map50 >= 0.85 else "\\u274c NO"} ({fp32_map50:.4f})')
print(f'  INT8 drop <= 2%?   {"\\u2705 YES" if drop_map50_pct <= 2.0 else "\\u274c NO"} ({drop_map50_pct:.1f}%)')
print(f'  INT8 model size:   {int8_size_mb:.2f} MB')""")

# =========================== CELL 7 HEADER ===========================
md("""---
## Cell 7: Save Artifacts to Kaggle Output""")

# =========================== CELL 7 ===========================
code("""# ============================================================
# Cell 7: Save Artifacts to Kaggle Output
# ============================================================

import zipfile

# Kaggle output directory (automatically available for download after run)
OUTPUT_DIR = Path('/kaggle/working')

# Package all artifacts into a single zip
zip_path = OUTPUT_DIR / 'aerosentinel_artifacts.zip'

artifacts = [
    (MODELS_DIR / 'best.pt', 'models/best.pt'),
    (MODELS_DIR / 'aerosentinel.onnx', 'models/aerosentinel.onnx'),
    (MODELS_DIR / 'aerosentinel_int8.onnx', 'models/aerosentinel_int8.onnx'),
    (MODELS_DIR / 'evaluation_results.md', 'models/evaluation_results.md'),
    (PROJECT_ROOT / 'synthetic_cracks_preview.png', 'demo/synthetic_cracks_preview.png'),
]

# Also include training plots if available
for plot_file in PROJECT_ROOT.glob('*.png'):
    if plot_file.name != 'synthetic_cracks_preview.png':
        artifacts.append((plot_file, f'plots/{plot_file.name}'))

print('Packaging artifacts...')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for src, arc_name in artifacts:
        if src.exists():
            zf.write(src, arc_name)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f'  \\u2705 {arc_name} ({size_mb:.2f} MB)')
        else:
            print(f'  \\u26a0\\ufe0f {arc_name} NOT FOUND, skipping')

# Also copy key files directly to /kaggle/working for easy individual download
for src, arc_name in artifacts:
    if src.exists():
        dst = OUTPUT_DIR / Path(arc_name).name
        shutil.copy2(src, dst)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f'\\n\\U0001f4e6 Archive: {zip_path} ({zip_size_mb:.2f} MB)')
print(f'\\n\\u2705 Done! All artifacts saved to /kaggle/working/')
print('To download: Click \"Save Version\" (top right) \\u2192 Save \\u2192 then go to Output tab.')""")


# =========================== BUILD NOTEBOOK ===========================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "cells": cells
}

# Fix escaped quotes in code cells (the triple-quote workaround)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        fixed_source = []
        for line in cell['source']:
            line = line.replace('\\"\\"\\"', '"""')
            fixed_source.append(line)
        cell['source'] = fixed_source

output_path = 'AeroSentinel_Training.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook written to {output_path}')
print(f'Total cells: {len(cells)}')
