import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model_path, amount=0.3):
    print(f"Loading model {model_path} for pruning...")
    model = YOLO(model_path)
    
    # Prune Conv2d layers in the PyTorch model
    parameters_to_prune = []
    for module_name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            
    print(f"Applying global unstructured pruning (L1 magnitude) at {amount*100}% sparsity...")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    print("Pruning applied and made permanent.")
    
    # Save the pruned model weights
    pruned_path = model_path.replace('.pt', '_pruned.pt')
    model.save(pruned_path)
    print(f"Pruned model saved to {pruned_path}")
    
    return pruned_path

def heal_model(pruned_path):
    print(f"Starting healing phase for {pruned_path}...")
    model = YOLO(pruned_path)
    # Train for a few epochs to heal the pruned weights
    model.train(
        data="split_crazing_only.yaml",
        epochs=15, # Brief healing phase
        batch=16,
        project="runs/detect",
        name="crazing_expert_9s_healed",
        exist_ok=True,
        lr0=0.001 # lower learning rate for healing
    )
    print("Healing complete!")

if __name__ == "__main__":
    best_weights = "runs/detect/runs/detect/crazing_expert_9s/weights/best.pt"
    # We prune 30% of the weights
    pruned_weights = prune_model(best_weights, amount=0.3)
    heal_model(pruned_weights)
