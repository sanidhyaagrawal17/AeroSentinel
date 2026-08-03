import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

model_path = '/home/ec2-user/runs/detect/tune/weights/best.pt'
print(f"Loading best GA-tuned model: {model_path}")
model = YOLO(model_path)

# Apply 30% Global Magnitude Pruning
print("Applying Global Magnitude Pruning (30%)...")
parameters_to_prune = []
# Ensure model is initialized properly by parsing through its PyTorch backend
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))

if not parameters_to_prune:
    print("No parameters found for pruning!")
    exit(1)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

# Permanently remove the pruning hooks and apply the 0 mask
for module, name in parameters_to_prune:
    prune.remove(module, name)

print("Pruning applied successfully! Healing for 20 epochs...")

# Fine-tune the pruned model for 20 epochs using the optimal GA hyperparameters!
model.train(
    data='split_full.yaml',
    epochs=20,
    optimizer='AdamW',
    cfg='/home/ec2-user/runs/detect/tune/best_hyperparameters.yaml', # use GA params
    device=0,
    project='runs/detect',
    name='pruned_crazing'
)

print("Healing complete! Saved to runs/detect/pruned_crazing/weights/best.pt")
