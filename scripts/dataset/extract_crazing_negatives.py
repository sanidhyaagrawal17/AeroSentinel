import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducible negative sampling
random.seed(42)

def has_crazing(label_path):
    if not os.path.exists(label_path):
        return False
    with open(label_path, 'r') as f:
        for line in f:
            class_id = line.split()[0]
            if class_id == '0':
                return True
    return False

def extract_dataset(src_dir, dest_dir, max_negatives):
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels'), exist_ok=True)
    
    src_images_dir = os.path.join(src_dir, 'images')
    src_labels_dir = os.path.join(src_dir, 'labels')
    
    if not os.path.exists(src_images_dir):
        return
        
    all_images = [f for f in os.listdir(src_images_dir) if f.endswith(('.jpg', '.png'))]
    crazing_images = []
    negative_images = []
    
    for img_file in all_images:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(src_labels_dir, label_file)
        
        if has_crazing(label_path):
            crazing_images.append(img_file)
        else:
            negative_images.append(img_file)
            
    print(f"Found {len(crazing_images)} Crazing images.")
    print(f"Found {len(negative_images)} Negative/Background images.")
    
    # Randomly sample negatives
    sampled_negatives = random.sample(negative_images, min(max_negatives, len(negative_images)))
    
    # Copy files
    for img_file in crazing_images + sampled_negatives:
        # Copy image
        shutil.copy(os.path.join(src_images_dir, img_file), os.path.join(dest_dir, 'images', img_file))
        # Copy label if it exists (for crazing, we only want to keep the Crazing labels. 
        # Wait, if an image has multiple defects including crazing, we should keep ONLY the crazing label for this expert!)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(src_labels_dir, label_file)
        dest_label_path = os.path.join(dest_dir, 'labels', label_file)
        
        if os.path.exists(src_label_path):
            with open(src_label_path, 'r') as src, open(dest_label_path, 'w') as dest:
                for line in src:
                    class_id = line.split()[0]
                    if class_id == '0': # ONLY KEEP CRAZING LABELS!
                        dest.write(line)
        else:
            # If no label exists (pure background), create an empty label file 
            # to signify to YOLO that this image is definitively empty.
            open(dest_label_path, 'w').close()
            
    print(f"Copied {len(crazing_images)} Crazing and {len(sampled_negatives)} Negatives to {dest_dir}")

def main():
    print("Extracting Crazing + Negative Sampling Dataset...")
    
    # We want ~300 negatives in train, and ~50 negatives in valid.
    extract_dataset("data/processed/train", "data/crazing_negatives/train", max_negatives=300)
    extract_dataset("data/processed/valid", "data/crazing_negatives/valid", max_negatives=50)
    extract_dataset("data/processed/test", "data/crazing_negatives/test", max_negatives=20)
    
if __name__ == "__main__":
    main()
