import os
from ultralytics import YOLO

def main():
    print("=== EXPERIMENT C: Synthetic Augmentation ===")
    # Train YOLO11n on the augmented dataset
    model = YOLO("models/yolo11n_aws_best.pt") # Start from anchor
    
    results = model.train(
        data="split_crazing_augmented.yaml",
        epochs=50,
        batch=32,
        project="runs/detect/bakeoff",
        name="expC_synthetic_augmentation",
        exist_ok=True
    )
    print("Experiment C Training Complete!")

if __name__ == "__main__":
    main()
