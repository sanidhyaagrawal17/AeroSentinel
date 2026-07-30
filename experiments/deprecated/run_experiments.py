
import os
import subprocess

def run():
    for i in range(1, 6):
        print(f"\n{'='*50}\nStarting Training for Strategy {i}\n{'='*50}")
        cmd = [
            r"venv\Scripts\yolo",
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
