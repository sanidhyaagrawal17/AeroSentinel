import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def export_to_onnx():
    config_path = Path(__file__).parent.parent / 'configs' / 'export_config.yaml'
    config = load_config(config_path)
    
    model_path = Path(__file__).parent.parent / 'models' / 'best.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print("Exporting to ONNX...")
    # Export the model
    # dynamic=True allows variable image sizes, simplify=True runs ONNX simplifier
    export_path = model.export(
        format=config['format'], 
        dynamic=config['dynamic'], 
        simplify=config['simplify'],
        imgsz=config['imgsz'],
        opset=config['opset'],
        half=config['half']
    )
    
    print(f"Model exported to {export_path}")
    
    # Rename and move to models directory
    target_path = Path(__file__).parent.parent / 'models' / 'aerosentinel.onnx'
    if Path(export_path).exists():
        shutil.copy2(export_path, target_path)
        os.unlink(export_path)
        print(f"Moved ONNX model to {target_path}")
    else:
        print(f"Export failed: {export_path} not found")

if __name__ == '__main__':
    export_to_onnx()
