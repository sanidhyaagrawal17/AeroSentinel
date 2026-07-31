import os
import pandas as pd
import glob

base_dir = r"."\runs\detect"
strategies = {
    1: "Randomized Background",
    2: "Procedural Textures",
    3: "Domain Randomization",
    4: "Synthetic Distractors",
    5: "Hard Negatives"
}

results = []

for i in range(1, 6):
    # Find the most recent experiment folder for this strategy
    pattern = os.path.join(base_dir, f"strat{i}_experiment*")
    folders = glob.glob(pattern)
    if not folders:
        print(f"No results found for Strategy {i}")
        continue
        
    # Sort by creation time to get the latest
    latest_folder = max(folders, key=os.path.getmtime)
    csv_path = os.path.join(latest_folder, "results.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Columns usually have leading spaces in YOLOv8
        df.columns = df.columns.str.strip()
        
        # Get the best mAP50 over the 20 epochs
        best_map50 = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
        
        results.append({
            'Strategy': i,
            'Name': strategies[i],
            'Best mAP@0.5': best_map50,
            'Best Epoch': best_epoch
        })

df_results = pd.DataFrame(results)
print("\n=== Real-World Validation mAP@0.5 Shootout ===")
print(df_results.to_string(index=False))
