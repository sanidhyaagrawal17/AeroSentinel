from roboflow import Roboflow
import os
from pathlib import Path

# ==========================================
# NEU-DET Dataset Downloader
# ==========================================
# The NEU-DET dataset is provided perfectly formatted for YOLOv8 via Roboflow Universe.
# 
# Instructions:
# 1. Create a free account at https://app.roboflow.com
# 2. Get your API key from Settings -> Workspace -> API Key
# 3. Replace 'YOUR_API_KEY_HERE' below with your actual key.
# 4. Run this script!
# ==========================================

API_KEY = os.getenv("ROBOFLOW_API_KEY", "your-api-key-here")

def download_dataset():
    if API_KEY == "your-api-key-here":
        print("Error: You must insert your Roboflow API key into this script first.")
        print("Get it for free at https://app.roboflow.com")
        return

    print("Authenticating with Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    print("Locating NEU-DET Object Detection project...")
    project = rf.workspace("park-sung-ho").project("neu-det-object-detection")
    version = project.version(1)
    
    # Download directly to the raw data directory
    download_dir = Path(__file__).parent.parent / 'data' / 'raw'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading YOLOv8 dataset to {download_dir}...")
    dataset = version.download("yolov8", location=str(download_dir))
    
    print("Download complete!")
    print("You can now run `python training/augmentation.py` to prepare it for aerospace.")

if __name__ == "__main__":
    download_dataset()
