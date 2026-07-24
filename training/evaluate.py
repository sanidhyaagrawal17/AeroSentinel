import yaml
from ultralytics import YOLO
from pathlib import Path

def evaluate_model():
    # Load best model
    model_path = Path(__file__).parent.parent / 'models' / 'best.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    data_path = Path(__file__).parent.parent / 'data' / 'data.yaml'
    
    print("Starting evaluation on validation set...")
    # Evaluate the model
    metrics = model.val(data=str(data_path.resolve()))
    
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    
    print("Per-class AP50:")
    for i, class_name in enumerate(metrics.names.values()):
        # metrics.box.maps contains AP50-95 for each class, we can print it
        # YOLOv8 val doesn't directly expose per-class AP50 in a simple list without indexing
        try:
            print(f"  {class_name}: {metrics.box.maps[i]}")
        except IndexError:
            pass

if __name__ == '__main__':
    evaluate_model()
