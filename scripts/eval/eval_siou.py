import os
import glob
from ultralytics import YOLO

def main():
    runs_dir = '/teamspace/studios/this_studio/runs/detect'
    dirs = glob.glob(os.path.join(runs_dir, 'custom_loss_siou*'))
    dirs.sort(key=os.path.getmtime)
    latest_dir = dirs[-1]
    
    model_path = os.path.join(latest_dir, 'weights', 'best.pt')
    print(f"Evaluating model: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(
        data='/teamspace/studios/this_studio/configs/real_neu_det.yaml',
        imgsz=320,
        split='val',
        device=0
    )
    print("mAP50 by class:")
    for i, name in enumerate(results.names.values()):
        print(f"{name}: {results.box.maps[i]:.4f}")

if __name__ == '__main__':
    main()
