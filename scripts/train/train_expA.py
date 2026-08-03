import os
from ultralytics import YOLO

def main():
    print("=== EXPERIMENT A: Hyperparameter Class-Balancing ===")
    # Train YOLO11n on the full dataset with image_weights and focal loss
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data="split_full.yaml",
        epochs=50,
        batch=32,
        project="runs/detect/bakeoff",
        name="expA_class_balancing",
        exist_ok=True,
        cls_pw=1.0 # Class weights power for handling class imbalance (1.0 = full inverse frequency)
    )
    print("Experiment A Training Complete!")

if __name__ == "__main__":
    main()
