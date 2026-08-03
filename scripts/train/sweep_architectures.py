import os
from ultralytics import YOLO

# Models to evaluate (Nano, Small, Tiny/Pico architectures from YOLOv8 to YOLO11)
models = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov9t.pt',
    'yolov9s.pt',
    'yolov10n.pt',
    'yolov10s.pt',
    'yolo11s.pt'
]

EPOCHS = 50

print("========================================")
print("STARTING CRAZING EXPERT ARCHITECTURAL SWEEP")
print(f"Models to evaluate: {len(models)}")
print(f"Epochs per model: {EPOCHS}")
print("========================================\n")

for m in models:
    name = m.replace('.pt', '')
    print(f"\n---> Training {name} for {EPOCHS} epochs...")
    try:
        model = YOLO(m)
        # Train on the cleaned dataset
        model.train(
            data='split_full.yaml',
            epochs=EPOCHS,
            imgsz=640,
            device=0,
            project='runs/detect/expert_sweep',
            name=name,
            exist_ok=True,
            verbose=False
        )
    except Exception as e:
        print(f"FAILED to train {name}: {e}")

print("\n========================================")
print("SWEEP COMPLETE! Check runs/detect/expert_sweep for results.")
print("========================================")
