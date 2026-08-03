import os
from ultralytics import YOLO

def run_evaluation(model_path, split_yaml="split_full.yaml", name="val_pruned_9s"):
    print(f"\n[Evaluating] {model_path} on Test Split...")
    model = YOLO(model_path)
    # Force evaluation on the 'test' split (which has the 180 real hold-out images)
    metrics = model.val(data=split_yaml, split='test', project="runs/detect", name=name)
    print(f"Results for {model_path}:")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    return metrics

if __name__ == "__main__":
    print("=== STRICT 180-IMAGE TEST SPLIT VALIDATION ===")
    
    # 1. The True Anchor (67% baseline)
    print("Loading the true 67% Anchor Model (YOLO11n)...")
    run_evaluation("models/yolo11n_aws_best.pt", name="anchor_yolo11n_eval")
    
    # 2. The Pruned & Healed YOLOv9s Crazing Expert
    pruned_healed_weights = "runs/detect/runs/detect/crazing_expert_9s_healed/weights/best.pt"
    if os.path.exists(pruned_healed_weights):
        print(f"\nLoading the Pruned & Healed YOLOv9s Crazing Expert...")
        run_evaluation(pruned_healed_weights, split_yaml="split_crazing_only.yaml", name="pruned_yolov9s_eval")
    else:
        print(f"ERROR: Could not find pruned weights at {pruned_healed_weights}")
