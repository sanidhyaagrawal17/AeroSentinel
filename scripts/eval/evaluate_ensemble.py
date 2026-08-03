import os

print("=== TRACK B: NATIVE ENSEMBLING ===")
print("Fusing top 3 architectures: YOLOv9t, YOLOv9s, YOLO11n")

models = [
    '/teamspace/studios/this_studio/runs/sweep_yolov9t/weights/best.pt',
    '/teamspace/studios/this_studio/runs/sweep_yolov9s/weights/best.pt',
    '/teamspace/studios/this_studio/runs/sweep_yolo11n/weights/best.pt'
]

# Ensure they exist
for m in models:
    if not os.path.exists(m):
        print(f"ERROR: Could not find {m}")
        exit(1)

model_str = ",".join(models)

# Use the Ultralytics CLI for guaranteed native ensembling support
print("\nEvaluating Ensemble on strict 169-image test split with TTA...")
cmd = f"yolo val model='{model_str}' data='/teamspace/studios/this_studio/split_neu_det.yaml' split=test augment=True imgsz=320 project='runs/detect' name='ensemble_val'"
os.system(cmd)

print("\n======================================")
print("TRACK B ENSEMBLE FINISHED.")
print("Please check the console output above for the final mAP50 score.")
print("======================================")
