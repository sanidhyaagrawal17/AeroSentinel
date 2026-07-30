import albumentations as A
import cv2
import numpy as np
import math
import random
from albumentations.core.transforms_interface import ImageOnlyTransform


class SyntheticCrackOverlay(ImageOnlyTransform):
    """
    Custom Albumentations transform that draws realistic synthetic crack
    patterns onto training images. Uses a random-walk polyline algorithm
    with branching to simulate surface cracks on metallic surfaces.
    
    This bridges the domain gap between the flat NEU steel dataset and
    real aerospace surfaces where hairline cracks are a critical defect.
    """
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
        """Draw a single crack path using random walk."""
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        angle = random.uniform(0, 2 * math.pi)

        steps = random.randint(*self.num_steps)
        thickness = random.randint(*self.thickness)
        color_val = random.randint(*self.color_range)
        color = (color_val, color_val, color_val)

        branch_points = []

        for step_i in range(steps):
            angle += math.radians(random.uniform(-self.angle_perturbation, self.angle_perturbation))
            step_len = random.uniform(*self.step_length)

            x_new = int(x + step_len * math.cos(angle))
            y_new = int(y + step_len * math.sin(angle))
            x_new = max(0, min(w - 1, x_new))
            y_new = max(0, min(h - 1, y_new))

            cv2.line(overlay, (x, y), (x_new, y_new), color, thickness)

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

        # Slight blur on crack lines to avoid pixel-sharp edges
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


def get_training_augmentation():
    """
    Returns an Albumentations pipeline designed to bridge the gap between
    the flat steel surfaces of the NEU dataset and curved, varied aerospace
    surfaces.
    """
    return A.Compose([
        # Random brightness and contrast to simulate different lighting conditions
        A.RandomBrightnessContrast(p=0.5),
        
        # Perspective warp to simulate viewing curved fuselage sections
        A.Perspective(scale=(0.05, 0.1), p=0.4),
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # to enhance local contrast, similar to what we do in inference
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        # Synthetic crack generation to bridge domain gap for hairline crack detection
        SyntheticCrackOverlay(p=0.4),
        
        # Basic geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        
        # Blur or noise to simulate camera sensor noise in dark hangars
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_validation_augmentation():
    """
    Validation pipeline typically just resizes, but YOLO handles this.
    We just return the CLAHE equivalent if we want to evaluate on enhanced images.
    """
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

import os
from pathlib import Path
import shutil

def augment_offline(data_yaml_path):
    """
    Applies augmentations to the dataset offline and saves them into an 'augmented' directory.
    This resolves the gap where YOLO doesn't natively hook into this specific Albumentations pipeline.
    """
    import yaml
    print(f"Loading data config: {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    base_dir = Path(data_yaml_path).parent
        
    splits = ['train', 'valid', 'test']
    count = 0
    
    visual_aug = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        SyntheticCrackOverlay(p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
    ])
    
    for split in splits:
        raw_images_dir = base_dir / 'raw' / split / 'images'
        raw_labels_dir = base_dir / 'raw' / split / 'labels'
        proc_images_dir = base_dir / 'processed' / split / 'images'
        proc_labels_dir = base_dir / 'processed' / split / 'labels'
        
        if not raw_images_dir.exists():
            continue
            
        proc_images_dir.mkdir(parents=True, exist_ok=True)
        proc_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing split: {split}...")
        for img_path in raw_images_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            if split == 'train':
                augmented = visual_aug(image=img)
                aug_img = augmented['image']
            else:
                aug_img = img # validation/test usually just use CLAHE, but for simplicity here we keep original or apply preprocessing at inference
                
            cv2.imwrite(str(proc_images_dir / img_path.name), aug_img)
            
            label_path = raw_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, proc_labels_dir / label_path.name)
                
            count += 1
            
    print(f"Offline augmentation complete. Processed {count} images across all splits.")
    
if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    aug = get_training_augmentation()
    print("Pipeline defined successfully.")
    
    # Example usage:
    data_yaml = Path(__file__).parent.parent / 'data' / 'data.yaml'
    augment_offline(data_yaml)
