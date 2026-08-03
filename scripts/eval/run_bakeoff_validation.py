import os
from ultralytics import YOLO

def evaluate_model(name, model_path, split_yaml="split_full.yaml"):
    print(f"\n{'='*50}")
    print(f"Evaluating {name}")
    print(f"Model Path: {model_path}")
    print(f"Dataset: {split_yaml}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
        
    try:
        model = YOLO(model_path)
        # We always evaluate on the strict hold-out 'test' split
        metrics = model.val(data=split_yaml, split='test', project="runs/detect/bakeoff_eval", name=name)
        
        # Extract the mAP50 for the Crazing class
        crazing_map50 = None
        if hasattr(metrics.box, 'ap50'):
            for i, class_idx in enumerate(metrics.ap_class_index):
                if class_idx == 0: # 0 is Crazing in all our datasets
                    crazing_map50 = metrics.box.ap50[i]
                    break
                
        print(f"\nRESULTS FOR {name}:")
        print(f"Overall mAP50:    {metrics.box.map50:.4f}")
        if crazing_map50 is not None:
            print(f"Crazing mAP50:    {crazing_map50:.4f}")
        else:
            print("Crazing mAP50:    Not found (0 instances?)")
            
        return crazing_map50
    except Exception as e:
        print(f"Evaluation failed for {name}: {e}")
        return None

def main():
    print("=== STARTING THE CRAZING REMEDIATION BAKEOFF VALIDATION ===")
    results = {}
    
    # 0. The Anchor
    results['Anchor (YOLO11n)'] = evaluate_model(
        "Anchor", 
        "models/yolo11n_aws_best.pt", 
        "split_full.yaml"
    )
    
    # 1. Experiment A (Class Balancing)
    results['Exp A (Class Balancing)'] = evaluate_model(
        "Experiment_A", 
        "runs/detect/runs/detect/bakeoff/expA_class_balancing/weights/best.pt", 
        "split_full.yaml"
    )
    
    # 2. Experiment B (Negative Sampling)
    # Must use the 1-class dataset YAML to prevent confusion matrix crash
    results['Exp B (Negative Sampling)'] = evaluate_model(
        "Experiment_B", 
        "runs/detect/runs/detect/bakeoff/expB_negative_sampling/weights/best.pt", 
        "split_crazing_negatives.yaml"
    )
    
    # 3. Experiment C (Synthetic Augmentation)
    results['Exp C (Synthetic Augmentation)'] = evaluate_model(
        "Experiment_C", 
        "runs/detect/runs/detect/bakeoff/expC_synthetic_augmentation/weights/best.pt", 
        "split_full.yaml" # Evaluate on the clean full test set, not augmented!
    )
    
    print("\n\n" + "#"*50)
    print("          FINAL CRAZING BAKEOFF LEADERBOARD")
    print("#"*50)
    
    # Sort results by crazing mAP50 descending
    valid_results = {k: v for k, v in results.items() if v is not None}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, map50) in enumerate(sorted_results):
        print(f"{i+1}. {name.ljust(30)} : {map50:.4f} mAP50")
    print("#"*50)

if __name__ == "__main__":
    main()
