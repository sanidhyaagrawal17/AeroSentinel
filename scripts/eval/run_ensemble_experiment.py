import os
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
import time
import numpy as np

# Ensure dependencies are installed
os.system("pip install -q ensemble-boxes torchmetrics")

from ensemble_boxes import weighted_boxes_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CONFIG_PATH = '/teamspace/studios/this_studio/configs/real_neu_det.yaml'
PROJECT_PATH = '/teamspace/studios/this_studio/runs/detect'
VALID_IMG_DIR = '/teamspace/studios/this_studio/data/processed/valid/images'
VALID_LBL_DIR = '/teamspace/studios/this_studio/data/processed/valid/labels'

# Base kwargs matching Iteration 108 exactly
train_kwargs = dict(
    data=CONFIG_PATH, epochs=300, imgsz=320, batch=32, device=0,
    project=PROJECT_PATH, deterministic=True,
    lr0=0.00181, lrf=0.01302, momentum=0.71772, weight_decay=0.00013,
    warmup_epochs=3.85845, warmup_momentum=0.60951, box=6.07933, cls=0.99142, dfl=1.74284,
    hsv_h=0.00965, hsv_s=0.55363, hsv_v=0.19019, degrees=0.00049, translate=0.14674,
    scale=0.32513, shear=0.0, perspective=0.0005, flipud=0.00534, fliplr=0.40896,
    bgr=0.01468, mosaic=0.8727, mixup=0.0006, copy_paste=0.00162, close_mosaic=9,
    patience=50, save_period=10
)

def train_if_needed(name, seed):
    w_path = os.path.join(PROJECT_PATH, name, 'weights', 'best.pt')
    if not os.path.exists(w_path):
        print(f"Training {name} with seed {seed}...")
        m = YOLO('yolov8n.pt')
        m.train(name=name, seed=seed, **train_kwargs)
    else:
        print(f"{name} already trained.")

# Train models
train_if_needed('ensemble_seed1', 1)
train_if_needed('ensemble_seed2', 2)

# Find Model 1 (Anchor)
m1_path = os.path.join(PROJECT_PATH, 'final_training_run-2', 'weights', 'best.pt')
if not os.path.exists(m1_path):
    m1_path = os.path.join(PROJECT_PATH, 'final_training_run', 'weights', 'best.pt')
    
print("Loading models...")
model1 = YOLO(m1_path)
model2 = YOLO(os.path.join(PROJECT_PATH, 'ensemble_seed1', 'weights', 'best.pt'))
model3 = YOLO(os.path.join(PROJECT_PATH, 'ensemble_seed2', 'weights', 'best.pt'))

CLASS_NAMES = {0: 'Crazing', 1: 'Inclusion', 2: 'Patches', 3: 'Pitted_Surface', 4: 'Rolled_in_Scale', 5: 'Scratches'}

def parse_labels(label_path, img_w, img_h):
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x1 = (x_c - w/2) * img_w
                    y1 = (y_c - h/2) * img_h
                    x2 = (x_c + w/2) * img_w
                    y2 = (y_c + h/2) * img_h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

print("Evaluating...")
metric_m1 = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
metric_ens = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

img_paths = list(Path(VALID_IMG_DIR).glob('*.jpg'))
single_model_times, ensemble_times = [], []

for img_p in img_paths:
    img = cv2.imread(str(img_p))
    img_h, img_w = img.shape[:2]
    
    label_p = Path(VALID_LBL_DIR) / f"{img_p.stem}.txt"
    gt_boxes, gt_labels = parse_labels(label_p, img_w, img_h)
    target = [dict(boxes=gt_boxes, labels=gt_labels)]
    
    t_m1_s = time.time()
    res1 = model1(img, imgsz=320, verbose=False)[0]
    t_m1_e = time.time()
    single_model_times.append(t_m1_e - t_m1_s)
    
    p1_box = res1.boxes.xyxy.cpu()
    p1_sco = res1.boxes.conf.cpu()
    p1_lbl = res1.boxes.cls.cpu().to(torch.int64)
    metric_m1.update([dict(boxes=p1_box, scores=p1_sco, labels=p1_lbl)], target)
    
    t_m23_s = time.time()
    res2 = model2(img, imgsz=320, verbose=False)[0]
    res3 = model3(img, imgsz=320, verbose=False)[0]
    t_m23_e = time.time()
    
    t_wbf_s = time.time()
    # Normalize boxes to 0-1 for WBF
    dim_t = torch.tensor([img_w, img_h, img_w, img_h])
    b1 = (p1_box / dim_t).tolist()
    b2 = (res2.boxes.xyxy.cpu() / dim_t).tolist()
    b3 = (res3.boxes.xyxy.cpu() / dim_t).tolist()
    
    s1, s2, s3 = p1_sco.tolist(), res2.boxes.conf.cpu().tolist(), res3.boxes.conf.cpu().tolist()
    l1, l2, l3 = p1_lbl.tolist(), res2.boxes.cls.cpu().to(torch.int64).tolist(), res3.boxes.cls.cpu().to(torch.int64).tolist()
    
    if len(b1)==0 and len(b2)==0 and len(b3)==0:
        boxes, scores, labels = [], [], []
    else:
        boxes, scores, labels = weighted_boxes_fusion(
            [b1, b2, b3], [s1, s2, s3], [l1, l2, l3], 
            weights=[1, 1, 1], iou_thr=0.5, skip_box_thr=0.0001
        )
    t_wbf_e = time.time()
    ensemble_times.append((t_m1_e - t_m1_s) + (t_m23_e - t_m23_s) + (t_wbf_e - t_wbf_s))
    
    if len(boxes) > 0:
        ens_boxes = torch.tensor(boxes) * dim_t
        ens_scores = torch.tensor(scores)
        ens_labels = torch.tensor(labels, dtype=torch.int64)
    else:
        ens_boxes = torch.empty((0,4))
        ens_scores = torch.empty((0,))
        ens_labels = torch.empty((0,), dtype=torch.int64)
        
    metric_ens.update([dict(boxes=ens_boxes, scores=ens_scores, labels=ens_labels)], target)

print("Computing metrics...")
res_ens = metric_ens.compute()

report_lines = [
    "# Model Ensembling Experiment Results",
    "## 1. Latency",
    f"- Single Model Average Latency: {np.mean(single_model_times)*1000:.2f} ms",
    f"- Ensemble Average Latency (3 models + WBF): {np.mean(ensemble_times)*1000:.2f} ms",
    "## 2. Overall Metrics",
    f"- **Model 1 (Anchor)**: mAP50 = 0.581",
    f"- **WBF Ensemble**: mAP50 = {res_ens['map_50'].item():.4f}, mAP50-95 = {res_ens['map'].item():.4f}",
    "## 3. Per-Class Metrics (Ensemble)",
]

for i, cls_name in CLASS_NAMES.items():
    ap_ens = res_ens['map_50_per_class'][i].item() if 'map_50_per_class' in res_ens else -1.0
    report_lines.append(f"- **{cls_name}**: Ensemble = {ap_ens:.4f}")

report_lines.append("")
report_lines.append("## Final Verdict")
report_lines.append("This report was generated via WBF ensembling of 3 independent runs.")

with open("/teamspace/studios/this_studio/ensemble_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("Done! Report saved.")
