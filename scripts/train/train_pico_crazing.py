import os
from ultralytics import YOLO

def main():
    # Load our winning Anchor (YOLO11n, 67% overall mAP50)
    print("Loading the 67% Anchor Model (YOLO11n) for Pico Fine-Tuning...")
    model = YOLO("models/yolo11n_aws_best.pt")
    
    # Train purely on the crazing_only dataset to eliminate background penalty
    # 275 total images (197 train, 50 valid, 28 test)
    print("Starting Pico Crazing Expert Training (30 Epochs)...")
    results = model.train(
        data="split_crazing_only.yaml",
        epochs=30,
        batch=16,
        project="runs/detect",
        name="pico_crazing_expert_11n",
        exist_ok=True,
        freeze=10, # Freeze backbone to retain generalized feature extraction and just fine-tune head on Crazing
        patience=15
    )
    
    print("Pico Crazing Expert training complete!")

if __name__ == "__main__":
    main()
