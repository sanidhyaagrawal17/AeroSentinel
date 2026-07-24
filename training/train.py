import yaml
from ultralytics import YOLO
from pathlib import Path
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'train_config.yaml'
    config = load_config(config_path)
    
    data_path = Path(__file__).parent.parent / 'data' / 'data.yaml'
    
    print(f"Loading base model: {config['model_name']}")
    model = YOLO(config['model_name'])
    
    # Normally, we would apply Albumentations here if using a custom Dataset class.
    # For Ultralytics YOLOv8, it has built-in augmentations, but to use our specific 
    # Albumentations pipeline (from augmentation.py), we'd either need to hook into 
    # the YOLO dataloader or pre-process the dataset offline.
    # For this PoC, we will let YOLO handle the augmentations built-in by mapping our
    # config parameters where possible, or assume pre-augmented data.
    
    print(f"Starting training on {data_path}...")
    
    # Train the model
    # Note: If running locally without GPU, this will be slow.
    results = model.train(
        data=str(data_path.resolve()),
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        device=config.get('device', 'cpu'),
        project=str(Path(__file__).parent / 'runs'),
        name='aerosentinel_train'
    )
    
    print("Training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    # Copy best model to models/ directory
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    target_path = Path(__file__).parent.parent / 'models' / 'best.pt'
    
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, target_path)
        print(f"Copied best model to {target_path}")

if __name__ == '__main__':
    train_model()
