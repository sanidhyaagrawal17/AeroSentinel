import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
DATA_DIR = Path('/teamspace/studios/this_studio/data/processed/train')
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'
MODEL_PATH = '/teamspace/studios/this_studio/runs/detect/aerosentinel_runs/yolov8m_teacher/weights/best.pt'
OUTPUT_REPORT = 'audit_report.json'
HTML_REPORT = 'audit_report.html'
VIS_DIR = Path('audit_visualizations')
VIS_DIR.mkdir(exist_ok=True)

CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in Scale", "Scratches"]

def compute_iou(box1, box2):
    # Format: [cx, cy, w, h] normalized
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def draw_boxes(img_path, gt_labels, pred_labels, output_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Draw GT (Green)
    for gt in gt_labels:
        cid, cx, cy, bw, bh = gt
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{CLASSES[int(cid)]}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw Pred (Red)
    for pred in pred_labels:
        cid, conf, cx, cy, bw, bh = pred
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"PR:{CLASSES[int(cid)]} {conf:.2f}", (x1, max(0, y2+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(str(output_path), img)

def main():
    print(f"Loading teacher model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    images = list(IMAGES_DIR.glob('*.jpg'))
    print(f"Found {len(images)} images in training set.")
    
    flags = []
    
    for img_path in images:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        
        gt_labels = []
        has_gt = label_path.exists()
        
        if has_gt:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid, cx, cy, bw, bh = map(float, parts)
                        gt_labels.append([cid, cx, cy, bw, bh])
                        
                        # Sanity checks
                        if bw <= 0.01 or bh <= 0.01:
                            flags.append({"image": img_path.name, "type": "degenerate_box", "reason": "Box too small"})
                        if bw >= 0.95 and bh >= 0.95:
                            flags.append({"image": img_path.name, "type": "oversized_box", "reason": "Box covers whole image"})
        else:
            flags.append({"image": img_path.name, "type": "missing_label_file", "reason": "No .txt file found"})

        # Model Inference
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        pred_labels = []
        for box in results[0].boxes:
            cid = int(box.cls[0].item())
            conf = box.conf[0].item()
            cx, cy, bw, bh = box.xywhn[0].tolist()
            pred_labels.append([cid, conf, cx, cy, bw, bh])

        # Logic Checks
        matched_gt = set()
        for p_idx, pred in enumerate(pred_labels):
            best_iou = 0
            best_gt_idx = -1
            
            for g_idx, gt in enumerate(gt_labels):
                iou = compute_iou(pred[2:], gt[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou > 0.4:
                matched_gt.add(best_gt_idx)
                gt = gt_labels[best_gt_idx]
                if int(pred[0]) != int(gt[0]):
                    if pred[1] > 0.5: # Confident mismatch
                        flags.append({
                            "image": img_path.name,
                            "type": "class_mismatch",
                            "reason": f"Pred: {CLASSES[int(pred[0])]} ({pred[1]:.2f}) vs GT: {CLASSES[int(gt[0])]}",
                            "gt": gt,
                            "pred": pred
                        })
            else:
                if pred[1] > 0.6: # High confidence ghost
                    flags.append({
                        "image": img_path.name,
                        "type": "ghost_prediction",
                        "reason": f"Confident prediction {CLASSES[int(pred[0])]} ({pred[1]:.2f}) with no GT overlap",
                        "pred": pred
                    })
                    
        for g_idx, gt in enumerate(gt_labels):
            if g_idx not in matched_gt:
                flags.append({
                    "image": img_path.name,
                    "type": "missing_detection",
                    "reason": f"GT {CLASSES[int(gt[0])]} was completely missed by model",
                    "gt": gt
                })

    # Save unique flagged images
    flagged_images = {}
    for f in flags:
        img_name = f["image"]
        if img_name not in flagged_images:
            flagged_images[img_name] = []
        flagged_images[img_name].append(f)
        
    print(f"Total images flagged: {len(flagged_images)}")
    
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(flagged_images, f, indent=4)
        
    # Generate Visuals for top 50
    html_content = "<html><body><h1>NEU-DET Audit Report</h1>"
    
    count = 0
    for img_name, img_flags in flagged_images.items():
        if count >= 50:
            break
            
        img_path = IMAGES_DIR / img_name
        vis_path = VIS_DIR / img_name
        
        # Reload GT and Pred for drawing (lazy way)
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        gt_l = []
        if label_path.exists():
            with open(label_path, 'r') as lf:
                for line in lf:
                    gt_l.append(list(map(float, line.strip().split())))
                    
        res = model.predict(source=str(img_path), conf=0.25, verbose=False)
        pr_l = []
        for box in res[0].boxes:
            pr_l.append([int(box.cls[0].item()), box.conf[0].item()] + box.xywhn[0].tolist())
            
        draw_boxes(img_path, gt_l, pr_l, vis_path)
        
        html_content += f"<h2>{img_name}</h2>"
        html_content += "<ul>"
        for flag in img_flags:
            html_content += f"<li><b>{flag['type']}:</b> {flag['reason']}</li>"
        html_content += "</ul>"
        html_content += f"<img src='audit_visualizations/{img_name}' width='400'/><br><hr>"
        count += 1

    html_content += "</body></html>"
    
    with open(HTML_REPORT, 'w') as f:
        f.write(html_content)
        
    print(f"Report saved to {OUTPUT_REPORT} and {HTML_REPORT}")

if __name__ == '__main__':
    main()
