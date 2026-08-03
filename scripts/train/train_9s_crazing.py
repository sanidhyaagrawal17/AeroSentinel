import os
from ultralytics import YOLO

def main():
    print("Loading YOLOv9s for Crazing Expert Fine-Tuning...")
    # Load YOLOv9s
    model = YOLO("yolov9c.pt") # Wait, yolov9 has yolov9c, yolov9e. YOLOv9 doesn't have a 's' officially? Ah, ultralytics has yolov9t, yolov9s, yolov9m, yolov9c, yolov9e.
    model = YOLO("yolov9s.pt")
    
    print("Starting YOLOv9s Crazing Expert Training (30 Epochs)...")
    results = model.train(
        data="split_crazing_only.yaml",
        epochs=30,
        batch=16,
        project="runs/detect",
        name="crazing_expert_9s",
        exist_ok=True,
        freeze=10, # Freeze backbone
        patience=15,
        lr0=0.01
    )
    
    print("YOLOv9s Crazing Expert training complete!")

if __name__ == "__main__":
    main()
