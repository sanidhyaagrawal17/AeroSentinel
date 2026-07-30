import os
import glob
import cv2
import numpy as np
import random
import argparse

def create_output_structure(base_dir):
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'labels', 'train'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_procedural_metal():
    # Stretched gaussian noise to look like brushed metal
    base = np.random.normal(128, 60, (640, 640)).astype(np.float32)
    # Stretch horizontally or vertically
    if random.random() > 0.5:
        base = cv2.GaussianBlur(base, (51, 1), 0)
    else:
        base = cv2.GaussianBlur(base, (1, 51), 0)
    
    # Add high frequency noise
    base += np.random.normal(0, 10, (640, 640))
    base = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

def get_background(strategy):
    bg_path = r"../.."\metal_bg.jpg"
    
    if strategy == 2:
        return generate_procedural_metal()
        
    elif strategy == 3:
        # Domain Randomization: solid colors or pure static
        r = random.random()
        if r < 0.33:
            # Solid random color
            color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
            return np.full((640, 640, 3), color, dtype=np.uint8)
        elif r < 0.66:
            # Pure RGB static
            return np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        else:
            # Checkerboard
            board = np.zeros((640, 640, 3), dtype=np.uint8)
            sz = random.randint(20, 100)
            for i in range(0, 640, sz):
                for j in range(0, 640, sz):
                    if ((i//sz) + (j//sz)) % 2 == 0:
                        board[i:i+sz, j:j+sz] = [255, 0, 255]
            return board

    # Load metal_bg for strats 1, 4, 5
    bg = cv2.imread(bg_path)
    if bg is None:
        bg = np.full((640, 640, 3), 150, dtype=np.uint8)
    bg = cv2.resize(bg, (640, 640))
    
    # Randomize it heavily (Strats 1, 4, 5)
    # Flips
    if random.random() > 0.5: bg = cv2.flip(bg, 0)
    if random.random() > 0.5: bg = cv2.flip(bg, 1)
    # Rotation (90 deg increments)
    k = random.randint(0, 3)
    bg = np.rot90(bg, k).copy() # Ensure contiguous array
    
    # Brightness/Contrast Jitter
    alpha = random.uniform(0.5, 1.5) # Contrast
    beta = random.uniform(-50, 50)   # Brightness
    bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)
    
    if strategy == 4:
        # Add distractors
        num_distractors = random.randint(2, 6)
        for _ in range(num_distractors):
            dtype = random.choice(['rivet', 'line', 'smudge'])
            x, y = random.randint(0, 640), random.randint(0, 640)
            if dtype == 'rivet':
                cv2.circle(bg, (x, y), random.randint(5, 15), (50, 50, 50), -1)
                cv2.circle(bg, (x, y), random.randint(5, 15), (100, 100, 100), 2)
            elif dtype == 'line':
                x2, y2 = x + random.randint(-100, 100), y + random.randint(-100, 100)
                cv2.line(bg, (x, y), (x2, y2), (40, 40, 40), random.randint(2, 6))
            elif dtype == 'smudge':
                sz = random.randint(20, 80)
                smudge = np.zeros((640, 640, 3), dtype=np.float32)
                cv2.circle(smudge, (x, y), sz, (1, 1, 1), -1)
                smudge = cv2.GaussianBlur(smudge, (51, 51), 0)
                bg = np.clip(bg.astype(np.float32) - (smudge * random.randint(30, 80)), 0, 255).astype(np.uint8)
                
    return bg

def get_defect_patches(img_dir, label_dir):
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    patches_by_class = {}
    
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        if not os.path.exists(label_path): continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            xmin = max(0, int((cx - bw / 2.0) * w))
            ymin = max(0, int((cy - bh / 2.0) * h))
            xmax = min(w, int((cx + bw / 2.0) * w))
            ymax = min(h, int((cy + bh / 2.0) * h))
            
            crop = img[ymin:ymax, xmin:xmax]
            if crop.shape[0] < 15 or crop.shape[1] < 15: continue
            
            if cls_id not in patches_by_class:
                patches_by_class[cls_id] = []
            patches_by_class[cls_id].append(crop)
            
    return patches_by_class

def blend_multiply(bg_roi, patch):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2BGR)
    
    median_val = np.median(gray_patch)
    if median_val == 0: median_val = 1
    
    normalized_patch = np.clip(patch * (255.0 / median_val), 0, 255).astype(np.float32)
    
    h, w = patch.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((X - max(cx, 1)) / max(cx, 1))**2 + ((Y - max(cy, 1)) / max(cy, 1))**2)
    
    fade_mask = np.ones((h, w), dtype=np.float32)
    fade_zone = dist > 0.4
    fade_mask[fade_zone] = np.exp(-6 * (dist[fade_zone] - 0.4)**2)
    fade_mask_3ch = np.dstack([fade_mask]*3)
    
    white_bg = np.full((h, w, 3), 255, dtype=np.float32)
    final_patch = (normalized_patch * fade_mask_3ch) + (white_bg * (1.0 - fade_mask_3ch))
    
    alpha = random.uniform(0.65, 0.90)
    multiplied = bg_roi.astype(np.float32) * (final_patch / 255.0)
    blended = (bg_roi.astype(np.float32) * (1 - alpha)) + (multiplied * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

def blend_patch_onto_surface(bg, patch, cls_id, target_cx, target_cy):
    bg_h, bg_w, _ = bg.shape
    patch_h, patch_w, _ = patch.shape

    scale = random.uniform(0.6, 1.4)
    new_w = max(15, int(patch_w * scale))
    new_h = max(15, int(patch_h * scale))
    resized_patch = cv2.resize(patch, (new_w, new_h))
    resized_patch = cv2.GaussianBlur(resized_patch, (3, 3), 0)

    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
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

    if cls_id in [0, 5]: # Crazing, Scratches
        gray_rot = cv2.cvtColor(rotated_patch, cv2.COLOR_BGR2GRAY)
        rotated_patch = cv2.cvtColor(gray_rot, cv2.COLOR_GRAY2BGR)
        
        mask = np.zeros(rotated_patch.shape, rotated_patch.dtype)
        Y, X = np.ogrid[:rot_h, :rot_w]
        cy, cx = rot_h / 2.0, rot_w / 2.0
        dist = np.sqrt(((X - cx) / max(cx,1))**2 + ((Y - cy) / max(cy,1))**2)
        mask[dist <= 0.95] = 255

        center = (int(target_cx), int(target_cy))
        try:
            blended = cv2.seamlessClone(rotated_patch, bg, mask, center, cv2.MIXED_CLONE)
            alpha = random.uniform(0.70, 1.0)
            bg[ymin:ymax, xmin:xmax] = cv2.addWeighted(
                bg[ymin:ymax, xmin:xmax], 1 - alpha, 
                blended[ymin:ymax, xmin:xmax], alpha, 0
            )
        except:
            pass
    else:
        bg_roi = bg[ymin:ymax, xmin:xmax]
        bg[ymin:ymax, xmin:xmax] = blend_multiply(bg_roi, rotated_patch)

    shrink_x = int(rot_w * 0.15)
    shrink_y = int(rot_h * 0.15)
    return bg, (xmin + shrink_x, ymin + shrink_y, xmax - shrink_x, ymax - shrink_y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=int, required=True, choices=[1,2,3,4,5])
    parser.add_argument('--num_images', type=int, default=200)
    args = parser.parse_args()
    
    base_dir = r"../.."\data\processed\train"
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    
    out_dir = rf"d:\AeroSentinel TATA AIG 2026\training\strat{args.strategy}"
    create_output_structure(out_dir)
    
    print("Loading patches...")
    patches_by_class = get_defect_patches(img_dir, lbl_dir)
    available_classes = list(patches_by_class.keys())
    
    print(f"Generating {args.num_images} images for Strategy {args.strategy}...")
    for i in range(args.num_images):
        bg = get_background(args.strategy)
        labels = []
        
        # Hard Negative Mining (Strategy 5): 20% empty backgrounds
        is_hard_negative = (args.strategy == 5 and random.random() < 0.2)
        
        if not is_hard_negative:
            num_defects = random.randint(1, 4)
            for _ in range(num_defects):
                cls_id = random.choice(available_classes)
                patch = random.choice(patches_by_class[cls_id])
                target_cx = random.randint(100, 540)
                target_cy = random.randint(100, 540)
                
                bg, box = blend_patch_onto_surface(bg, patch, cls_id, target_cx, target_cy)
                if box is not None:
                    xmin, ymin, xmax, ymax = box
                    labels.append((cls_id, xmin, ymin, xmax, ymax))
                
        # Save image
        img_filename = f"synth_strat{args.strategy}_{i:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, 'images', 'train', img_filename), bg)
        
        # Save label
        with open(os.path.join(out_dir, 'labels', 'train', f"synth_strat{args.strategy}_{i:04d}.txt"), 'w') as f:
            for cls_id, x1, y1, x2, y2 in labels:
                x_center = ((x1 + x2) / 2.0) / 640.0
                y_center = ((y1 + y2) / 2.0) / 640.0
                w_box = (x2 - x1) / 640.0
                h_box = (y2 - y1) / 640.0
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")
                
    print(f"Dataset generated at {out_dir}")

if __name__ == "__main__":
    main()
