import os
from ultralytics import YOLO

def main():
    # 1. Initialize the custom BiFPN model architecture (Random Weights / No COCO)
    # Using the yaml ensures we load the BiFPN structure.
    print("Initializing YOLOv8n with BiFPN architecture...")
    model = YOLO('yolov8n-bifpn.yaml') 
    
    try:
        model.load("yolov8n.pt")
        print("Loaded pretrained backbone weights from yolov8n.pt")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")

    # 2. Define the exact GA-Tuned Hyperparameters optimized for Batch 16
    ga_hyperparameters = {
        'lr0': 0.00181,
        'lrf': 0.0105,
        'momentum': 0.914,
        'weight_decay': 0.00045,
        'warmup_epochs': 3.85,
        'warmup_momentum': 0.75,
        'box': 7.15,
        'cls': 0.58,
        'dfl': 1.25,
        'hsv_h': 0.013,
        'hsv_s': 0.65,
        'hsv_v': 0.38,
        'degrees': 0.0,
        'translate': 0.09,
        'scale': 0.45,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.85,
        'mixup': 0.0,
        'copy_paste': 0.0
    }

    # 3. Execute the Training Run
    print("Starting True BiFPN training run with batch=16...")
    results = model.train(
        data='/teamspace/studios/this_studio/configs/real_neu_det.yaml', # Points to the 1259-image split
        epochs=300,                     # Standard run length
        patience=50,                    # Standard early stopping
        batch=16,                       # CRITICAL FIX: Restored to GA optimized size
        imgsz=320,                      # Maintained for apples-to-apples baseline comparison
        name='true_bifpn_ciou_batch16', # Distinct run name
        device=0,                       # Force GPU
        optimizer='AdamW',              # Fix: Use GA's optimizer so lr0/momentum aren't overwritten
        seed=0,                         # Reproducibility
        **ga_hyperparameters            # Inject the exact tuning dictionary
    )
    
    print(f"Training completed successfully. Best weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
