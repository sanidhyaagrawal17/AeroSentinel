import os
from ultralytics import YOLO

def main():
    print("=== STARTING YOLO11N CRAZING EXPERT TRAINING ===")
    
    # We use YOLO11n as the base since it performed best on the bakeoff
    model = YOLO("yolo11n.pt") 

    results = model.train(
        data="split_crazing_only.yaml", 
        epochs=50, 
        imgsz=640,
        batch=32,
        project="runs/detect/expert",
        name="yolo11n_crazing_expert",
        exist_ok=True,
    )
    
    print("Training Complete!")
    
if __name__ == "__main__":
    main()
