import os
import yaml

base_dir = r".""
config_dir = os.path.join(base_dir, "configs")
os.makedirs(config_dir, exist_ok=True)

class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']

for i in range(1, 6):
    config = {
        'path': base_dir,
        'train': f'training/strat{i}/images/train',
        # CRITICAL: Validate on the real NEU-DET data
        'val': 'data/processed/valid/images', 
        'nc': len(class_names),
        'names': class_names
    }
    
    config_path = os.path.join(config_dir, f"strat{i}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

print("Created 5 YAML configurations.")

# Write run_experiments.py
runner_code = """
import os
import subprocess

def run():
    for i in range(1, 6):
        print(f"\\n{'='*50}\\nStarting Training for Strategy {i}\\n{'='*50}")
        cmd = [
            r"venv\\Scripts\\yolo",
            "detect", "train",
            "model=yolov8n.pt",
            f"data=configs/strat{i}.yaml",
            "epochs=20",
            "imgsz=640",
            "batch=16",
            f"name=strat{i}_experiment"
        ]
        subprocess.run(cmd)

if __name__ == '__main__':
    run()
"""

with open(os.path.join(base_dir, "run_experiments.py"), 'w') as f:
    f.write(runner_code)
print("Created run_experiments.py")
