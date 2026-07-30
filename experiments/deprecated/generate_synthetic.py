import os
import cv2
import numpy as np
import random
import math

def create_yolo_directories(base_dir):
    """Creates the standard YOLO dataset folder structure, plus a visualize folder."""
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val'),
        os.path.join(base_dir, 'visualize') # Added for sanity check
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_aluminum_background(width=640, height=640):
    """Procedurally generates a brushed metal/aluminum aircraft panel."""
    base_color = random.randint(140, 200)
    img = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add textural noise
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply directional blur for brushed metal effect
    k_size = random.choice([15, 21, 31])
    kernel = np.zeros((k_size, k_size))
    if random.random() > 0.5:
        kernel[k_size//2, :] = np.ones(k_size) # Horizontal brush
    else:
        kernel[:, k_size//2] = np.ones(k_size) # Vertical brush
    kernel /= k_size
    img = cv2.filter2D(img, -1, kernel)
    
    # 50% chance to add structural seams and rivets
    if random.random() > 0.5:
        seam_x = random.randint(100, width - 100)
        cv2.line(img, (seam_x, 0), (seam_x, height), (70, 70, 70), 2)
        for y in range(20, height, 40):
            cv2.circle(img, (seam_x - 12, y), 3, (90, 90, 90), -1)
            cv2.circle(img, (seam_x - 12, y), 3, (200, 200, 200), 1)
            
    return img

def generate_procedural_dataset(num_images=5, output_dir="AeroSentinel_Synthetic"):
    """Generates images, YOLO annotations, and drawn visualization frames."""
    create_yolo_directories(output_dir)
    
    for i in range(num_images):
        # 80/20 Train/Val Split
        split = 'train' if i < num_images * 0.8 else 'val'
        
        width, height = 640, 640
        img = generate_aluminum_background(width, height)
        vis_img = img.copy() # Copy to draw boxes for visual inspection
        labels = []
        
        # Determine number of defects on this panel (1 to 3)
        num_defects = random.randint(1, 3)
        
        for _ in range(num_defects):
            defect_type = random.choice([0, 1]) # 0: Crack, 1: Corrosion
            
            if defect_type == 0:
                # --- SYNTHETIC CRACK GENERATION ---
                cx, cy = random.randint(80, width - 80), random.randint(80, height - 80)
                min_x, min_y, max_x, max_y = cx, cy, cx, cy
                
                # Random walk math for jagged lines
                for _ in range(random.randint(10, 25)):
                    angle = random.uniform(0, 2 * math.pi)
                    length = random.randint(10, 35)
                    nx = np.clip(int(cx + length * math.cos(angle)), 0, width - 1)
                    ny = np.clip(int(cy + length * math.sin(angle)), 0, height - 1)
                    
                    thickness = random.randint(1, 3)
                    cv2.line(img, (cx, cy), (nx, ny), (20, 20, 20), thickness)
                    cv2.line(vis_img, (cx, cy), (nx, ny), (20, 20, 20), thickness) # Draw on vis_img too
                    
                    min_x, max_x = min(min_x, cx, nx), max(max_x, cx, nx)
                    min_y, max_y = min(min_y, cy, ny), max(max_y, cy, ny)
                    cx, cy = nx, ny
                    
                labels.append((0, min_x, min_y, max_x, max_y))
                
            else:
                # --- SYNTHETIC CORROSION GENERATION ---
                cx, cy = random.randint(80, width - 80), random.randint(80, height - 80)
                base_radius = random.randint(30, 75)
                
                # Mask out an irregular pitted area
                for y in range(max(0, cy - base_radius), min(height, cy + base_radius)):
                    for x in range(max(0, cx - base_radius), min(width, cx + base_radius)):
                        dist = math.hypot(x - cx, y - cy)
                        # Add high-frequency noise to the radius to make it jagged
                        if dist + random.randint(-15, 20) < base_radius:
                            # Apply rust/pitted BGR coloring
                            if random.random() > 0.3:
                                color = [random.randint(20, 60), random.randint(50, 90), random.randint(100, 150)]
                            else:
                                color = [random.randint(40, 70), random.randint(40, 70), random.randint(40, 70)]
                            img[y, x] = color
                            vis_img[y, x] = color # Draw on vis_img too
                                
                min_x, max_x = max(0, cx - base_radius), min(width - 1, cx + base_radius)
                min_y, max_y = max(0, cy - base_radius), min(height - 1, cy + base_radius)
                labels.append((1, min_x, min_y, max_x, max_y))
                
        # --- SAVE IMAGE AND AUTOMATIC LABELS ---
        img_filename = f"synth_{i:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, 'images', split, img_filename), img)
        
        with open(os.path.join(output_dir, 'labels', split, f"synth_{i:05d}.txt"), 'w') as f:
            for cls_id, x1, y1, x2, y2 in labels:
                # Add a 5px safety padding to the bounding box
                x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
                x2, y2 = min(width, x2 + 5), min(height, y2 + 5)
                
                # Convert to YOLO Normalized Format
                x_center = ((x1 + x2) / 2.0) / width
                y_center = ((y1 + y2) / 2.0) / height
                w_box = (x2 - x1) / width
                h_box = (y2 - y1) / height
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")
                
                # --- VISUALIZATION DRAWING ---
                # Draw the actual bounding box so you can visually verify it's accurate
                box_color = (0, 255, 0) if cls_id == 0 else (0, 0, 255) # Green for crack, Red for corrosion
                label_name = "Crack" if cls_id == 0 else "Corrosion"
                
                # OpenCV requires integer coordinates for drawing rectangles
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.putText(vis_img, label_name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # Save the visualization image
        vis_filename = f"vis_{i:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, 'visualize', vis_filename), vis_img)
                
    print(f"[SUCCESS] Generated {num_images} fully automated synthetic images in '{output_dir}'")
    print(f"[INFO] Check the '{output_dir}/visualize' folder to visually inspect the bounding boxes!")

if __name__ == "__main__":
    # Generating a small batch of 5 for sanity check
    generate_procedural_dataset(5)
