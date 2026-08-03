import os
import shutil
import json
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path('/teamspace/studios/this_studio/data/processed/train')
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'
QUARANTINE_DIR = Path('/teamspace/studios/this_studio/data/processed/train_quarantine')
MODEL_PATH = '/teamspace/studios/this_studio/runs/detect/aerosentinel_runs/yolov8m_teacher/weights/best.pt'

def main():
    # 1. Restore the quarantined files
    print("Restoring quarantined files back to the training set...")
    restored_count = 0
    if QUARANTINE_DIR.exists():
        for q_img in (QUARANTINE_DIR / 'images').glob('*.jpg'):
            shutil.move(str(q_img), str(IMAGES_DIR / q_img.name))
            
            q_label = (QUARANTINE_DIR / 'labels' / q_img.name.replace('.jpg', '.txt'))
            if q_label.exists():
                shutil.move(str(q_label), str(LABELS_DIR / q_label.name))
            restored_count += 1
            
    print(f"Restored {restored_count} images.")
    
    # 2. Relabel the flagged images using the teacher model
    with open('audit_report.json', 'r') as f:
        report = json.load(f)
        
    flagged_images = list(report.keys())
    print(f"Loading teacher model to relabel {len(flagged_images)} flagged images...")
    
    model = YOLO(MODEL_PATH)
    
    relabel_count = 0
    for img_name in flagged_images:
        img_path = IMAGES_DIR / img_name
        label_path = LABELS_DIR / img_name.replace('.jpg', '.txt')
        
        if img_path.exists():
            # Predict
            results = model.predict(source=str(img_path), conf=0.25, verbose=False)
            
            # Write new label file
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    cid = int(box.cls[0].item())
                    cx, cy, bw, bh = box.xywhn[0].tolist()
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    
            relabel_count += 1
            
    print(f"Successfully auto-relabeled {relabel_count} images.")

if __name__ == '__main__':
    main()
