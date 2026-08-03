import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path('/teamspace/studios/this_studio/data/processed/train')
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'
QUARANTINE_DIR = Path('/teamspace/studios/this_studio/data/processed/train_quarantine')

def main():
    with open('audit_report.json', 'r') as f:
        report = json.load(f)
        
    flagged_images = list(report.keys())
    print(f"Loaded {len(flagged_images)} flagged images from report.")
    
    # Create quarantine folders
    (QUARANTINE_DIR / 'images').mkdir(parents=True, exist_ok=True)
    (QUARANTINE_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    for img_name in flagged_images:
        img_path = IMAGES_DIR / img_name
        label_name = img_name.replace('.jpg', '.txt')
        label_path = LABELS_DIR / label_name
        
        if img_path.exists():
            shutil.move(str(img_path), str(QUARANTINE_DIR / 'images' / img_name))
            if label_path.exists():
                shutil.move(str(label_path), str(QUARANTINE_DIR / 'labels' / label_name))
            moved_count += 1
            
    print(f"Moved {moved_count} images (and their labels) to {QUARANTINE_DIR}")
    
    remaining_images = list(IMAGES_DIR.glob('*.jpg'))
    print(f"Remaining clean images in training set: {len(remaining_images)}")

if __name__ == '__main__':
    main()
