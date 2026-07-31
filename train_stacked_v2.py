import os
from ultralytics import YOLO

def main():
    print("Initializing YOLOv8n with Stacked V2 (BiFPN + CBAM) architecture...")
    model = YOLO('yolov8n-bifpn-cbam.yaml')
    
    # We must enforce batch=16 (stable) and GA-tuned hyperparameters.
    # WIoU is reverted, we use default CIoU.
    print("Starting training with GA-tuned hyperparameters and batch=16...")
    results = model.train(
        data='/teamspace/studios/this_studio/configs/real_neu_det.yaml',
        epochs=300,
        batch=16,
        imgsz=320,
        device=0,
        project='aerosentinel_runs',
        name='stacked_v2_run',
        optimizer='AdamW',
        lr0=0.00181,
        lrf=0.0102,
        momentum=0.914,
        weight_decay=0.00045,
        warmup_epochs=2.87,
        warmup_momentum=0.71,
        box=7.02,
        cls=0.48,
        dfl=1.35,
        hsv_h=0.0138,
        hsv_s=0.664,
        hsv_v=0.464,
        degrees=0.0,
        translate=0.104,
        scale=0.491,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.95,
        mixup=0.081,
        copy_paste=0.063,
        patience=50
    )
    print("Training complete. Results saved to aerosentinel_runs/stacked_v2_run")

if __name__ == "__main__":
    main()
