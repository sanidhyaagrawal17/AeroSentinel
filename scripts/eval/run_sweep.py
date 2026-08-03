import os
from ultralytics import YOLO

# Define the models to sweep
models_to_test = [
    # Nano / Tiny tier
    'yolov8n.pt',
    'yolov9t.pt',  # v9 uses 't' for tiny instead of 'n'
    'yolov10n.pt',
    'yolo11n.pt',
    
    # Small tier
    'yolov8s.pt',
    'yolov9s.pt',
    'yolov10s.pt',
    'yolo11s.pt'
]

results = {}

for model_name in models_to_test:
    run_name = f"sweep_{model_name.replace('.pt', '')}"
    print(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}\n", flush=True)
    
    try:
        model = YOLO(model_name)
        
        # Train
        model.train(
            data='/teamspace/studios/this_studio/split_neu_det.yaml',
            epochs=100,      # Max epochs
            patience=30,     # Stop early if no improvement
            imgsz=320,       # Fast and historically optimal
            batch=64,        # Maximize batchnorm
            project='/teamspace/studios/this_studio/runs',
            name=run_name,
            exist_ok=True
        )
        
        # Final Evaluation with Test-Time Augmentation (TTA) on strict test set
        print(f"\nEvaluating {model_name} with TTA on Test Set...", flush=True)
        best_weights = f"/teamspace/studios/this_studio/runs/{run_name}/weights/best.pt"
        if os.path.exists(best_weights):
            eval_model = YOLO(best_weights)
            metrics = eval_model.val(
                data='/teamspace/studios/this_studio/split_neu_det.yaml', 
                split='test',
                augment=True  # ENABLE TEST TIME AUGMENTATION
            )
            map50 = metrics.box.map50
            results[model_name] = map50
            print(f">>> {model_name} TTA Test mAP50: {map50:.4f}", flush=True)
        else:
            print(f"Failed to find best.pt for {model_name}")
            
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Print final summary
print("\n\n" + "="*50)
print("SWEEP RESULTS (TTA Test mAP50)")
print("="*50)
for m, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{m:<15}: {score:.4f}")

# Save to file
with open('/teamspace/studios/this_studio/sweep_summary.txt', 'w') as f:
    f.write("SWEEP RESULTS (TTA Test mAP50)\n")
    for m, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        f.write(f"{m:<15}: {score:.4f}\n")
