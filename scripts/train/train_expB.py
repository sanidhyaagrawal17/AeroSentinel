import os
from ultralytics import YOLO

def main():
    print("=== EXPERIMENT B: Data-Centric Negative Sampling ===")
    # Train YOLO11n on the custom negative sampled dataset
    model = YOLO("models/yolo11n_aws_best.pt") # Start from anchor for faster convergence
    
    results = model.train(
        data="split_crazing_negatives.yaml",
        epochs=50,
        batch=32,
        project="runs/detect/bakeoff",
        name="expB_negative_sampling",
        exist_ok=True
    )
    print("Experiment B Training Complete!")

if __name__ == "__main__":
    main()
