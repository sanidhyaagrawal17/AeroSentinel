import os
import glob
import cv2
import numpy as np
import random
import math

def create_output_structure(base_dir):
    """Creates the standard YOLO dataset folder structure and visualization directory."""
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val'),
        os.path.join(base_dir, 'visualize')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_neu_defect_patches(neu_img_dir, neu_label_dir):
    """
    Scans the NEU-DET dataset directory, reads images and YOLO bounding boxes,
    and extracts real crop patches for each class.
    """
    patches_by_class = {}
    img_files = glob.glob(os.path.join(neu_img_dir, "*.jpg")) + glob.glob(os.path.join(neu_img_dir, "*.png"))
    
    print(f"[INFO] Scanning {len(img_files)} NEU-DET source images in {neu_img_dir}...")
    
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(neu_label_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            xmin = max(0, int((cx - bw / 2.0) * w))
            ymin = max(0, int((cy - bh / 2.0) * h))
            xmax = min(w, int((cx + bw / 2.0) * w))
            ymax = min(h, int((cy + bh / 2.0) * h))
            
            crop = img[ymin:ymax, xmin:xmax]
            
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
                
            if cls_id not in patches_by_class:
                patches_by_class[cls_id] = []
            patches_by_class[cls_id].append(crop)
            
    total_patches = sum(len(v) for v in patches_by_class.values())
    print(f"[SUCCESS] Extracted {total_patches} authentic defect patches across {len(patches_by_class)} classes.")
    return patches_by_class

def generate_metal_background(width=640, height=640):
    """Procedurally generates a high-quality brushed metal/aluminum panel background."""
    base_color = random.randint(150, 195)
    img = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    k_size = random.choice([21, 31, 41])
    kernel = np.zeros((k_size, k_size))
    if random.random() > 0.5:
        kernel[k_size // 2, :] = np.ones(k_size)
    else:
        kernel[:, k_size // 2] = np.ones(k_size)
    kernel /= k_size
    img = cv2.filter2D(img, -1, kernel)
    
    if random.random() > 0.4:
        seam_x = random.randint(80, width - 80)
        cv2.line(img, (seam_x, 0), (seam_x, height), (80, 80, 80), 2)
        for y in range(25, height, 45):
            cv2.circle(img, (seam_x - 10, y), 3, (100, 100, 100), -1)
            cv2.circle(img, (seam_x - 10, y), 3, (210, 210, 210), 1)
            
    return img

def blend_patch_onto_surface(bg, patch, target_cx, target_cy):
    """
    Applies random scale/rotation and uses a Luminance-Normalized Multiply Blend.
    """
    bg_h, bg_w, _ = bg.shape
    patch_h, patch_w, _ = patch.shape
    
    scale = random.uniform(0.6, 1.4)
    new_w = max(15, int(patch_w * scale))
    new_h = max(15, int(patch_h * scale))
    resized_patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    rot_w = int((new_h * sin) + (new_w * cos))
    rot_h = int((new_h * cos) + (new_w * sin))
    
    M[0, 2] += (rot_w / 2) - (new_w / 2)
    M[1, 2] += (rot_h / 2) - (new_h / 2)
    
    rotated_patch = cv2.warpAffine(resized_patch, M, (rot_w, rot_h), borderMode=cv2.BORDER_REPLICATE)
    
    xmin = max(0, int(target_cx - rot_w / 2))
    ymin = max(0, int(target_cy - rot_h / 2))
    xmax = min(bg_w, xmin + rot_w)
    ymax = min(bg_h, ymin + rot_h)
    
    actual_w = xmax - xmin
    actual_h = ymax - ymin
    
    if actual_w <= 0 or actual_h <= 0 or actual_w != rot_w or actual_h != rot_h:
        return bg, None
        
    bg_roi = bg[ymin:ymax, xmin:xmax]
    
    # 1. Grayscale Conversion
    gray_patch = cv2.cvtColor(rotated_patch, cv2.COLOR_BGR2GRAY)
    
    # 2. Find Background Luminance
    median_val = np.median(gray_patch)
    if median_val == 0:
        median_val = 1.0
        
    # 3. Normalize to White
    normalized_patch = np.clip(gray_patch.astype(np.float32) * (255.0 / median_val), 0, 255)
    
    # 4. Edge Fade to White
    Y, X = np.ogrid[:rot_h, :rot_w]
    center_y, center_x = rot_h / 2.0, rot_w / 2.0
    Y_norm = (Y - center_y) / (rot_h / 2.0)
    X_norm = (X - center_x) / (rot_w / 2.0)
    dist = np.sqrt(X_norm**2 + Y_norm**2)
    
    # Apply fade on outer 20% (dist > 0.8)
    fade_start = 0.8
    fade_mask = np.zeros((rot_h, rot_w), dtype=np.float32)
    fade_zone = dist > fade_start
    
    # Linear fade to 1 at the extreme edges
    fade_mask[fade_zone] = np.clip((dist[fade_zone] - fade_start) / (1.0 - fade_start), 0, 1)
    
    # Fade edges to pure white (255)
    faded_patch = (normalized_patch * (1.0 - fade_mask)) + (255.0 * fade_mask)
    faded_patch_3ch = np.dstack([faded_patch] * 3) / 255.0
    
    # 5. Multiply Blend
    bg_roi_f = bg_roi.astype(np.float32)
    blended = bg_roi_f * faded_patch_3ch
    
    bg[ymin:ymax, xmin:xmax] = np.clip(blended, 0, 255).astype(np.uint8)
    
    # 6. Update Bounding Box: Tighten by 15%
    shrink_x = int(rot_w * 0.15)
    shrink_y = int(rot_h * 0.15)
    xmin_tight = xmin + shrink_x
    ymin_tight = ymin + shrink_y
    xmax_tight = xmax - shrink_x
    ymax_tight = ymax - shrink_y
    
    return bg, (xmin_tight, ymin_tight, xmax_tight, ymax_tight)

def generate_authentic_dataset(neu_img_dir, neu_label_dir, num_images=5, output_dir="AeroSentinel_Authentic"):
    """Orchestrates authentic dataset synthesis using real cropped defects."""
    create_output_structure(output_dir)
    patches_by_class = load_neu_defect_patches(neu_img_dir, neu_label_dir)
    
    if not patches_by_class:
        print("[ERROR] No valid defect patches found. Check your input dataset paths!")
        return

    available_classes = list(patches_by_class.keys())
    
    for i in range(num_images):
        split = 'train' if i < num_images * 0.8 else 'val'
        width, height = 640, 640
        
        bg = generate_metal_background(width, height)
        vis_img = bg.copy()
        labels = []
        
        num_defects = random.randint(1, 3)
        
        for _ in range(num_defects):
            cls_id = random.choice(available_classes)
            patch = random.choice(patches_by_class[cls_id])
            
            target_cx = random.randint(100, width - 100)
            target_cy = random.randint(100, height - 100)
            
            bg, box = blend_patch_onto_surface(bg, patch, target_cx, target_cy)
            
            if box is not None:
                xmin, ymin, xmax, ymax = box
                labels.append((cls_id, xmin, ymin, xmax, ymax))
                
                box_color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)
                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), box_color, 2)
                cv2.putText(vis_img, f"Class {cls_id}", (xmin, max(15, ymin - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
        
        img_filename = f"auth_synth_{i:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, 'images', split, img_filename), bg)
        
        with open(os.path.join(output_dir, 'labels', split, f"auth_synth_{i:05d}.txt"), 'w') as f:
            for cls_id, x1, y1, x2, y2 in labels:
                x_center = ((x1 + x2) / 2.0) / width
                y_center = ((y1 + y2) / 2.0) / height
                w_box = (x2 - x1) / width
                h_box = (y2 - y1) / height
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")
                
        cv2.imwrite(os.path.join(output_dir, 'visualize', f"vis_{i:05d}.jpg"), vis_img)
        
    print(f"[SUCCESS] Successfully generated {num_images} authentic synthetic samples in '{output_dir}'.")
    print(f"[INFO] Check '{output_dir}/visualize' to verify the Luminance-Normalized Multiply Blending quality!")

if __name__ == "__main__":
    NEU_IMAGES_PATH = r".."\data\raw\train\images"
    NEU_LABELS_PATH = r".."\data\raw\train\labels"
    
    generate_authentic_dataset(NEU_IMAGES_PATH, NEU_LABELS_PATH, num_images=5)
