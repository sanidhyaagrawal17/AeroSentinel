from ultralytics import YOLO
from pathlib import Path

def main():
    # Load the YOLOv8 nano model
    model = YOLO('yolov8n.pt') 

    config_path = str(Path(__file__).parent.parent / 'configs' / 'real_neu_det.yaml')
    project_path = str(Path(__file__).parent.parent / 'runs' / 'detect')

    # Start training using the best hyperparameters found by the GA at Iteration 108
    results = model.train(
        data=config_path,
        epochs=300,
        imgsz=320,
        batch=32,
        device=0,
        project=project_path,
        name='final_training_run',
        # Apply the exact best hyperparameters
        lr0=0.00181,
        lrf=0.01302,
        momentum=0.71772,
        weight_decay=0.00013,
        warmup_epochs=3.85845,
        warmup_momentum=0.60951,
        box=6.07933,
        cls=0.99142,
        dfl=1.74284,
        hsv_h=0.00965,
        hsv_s=0.55363,
        hsv_v=0.19019,
        degrees=0.00049,
        translate=0.14674,
        scale=0.32513,
        shear=0.0,
        perspective=0.0005,
        flipud=0.00534,
        fliplr=0.40896,
        bgr=0.01468,
        mosaic=0.8727,
        mixup=0.0006,
        copy_paste=0.00162,
        close_mosaic=9,
        # Early stopping and saving
        patience=50,
        save_period=10
    )

if __name__ == '__main__':
    main()
