import csv, os

runs_base = '/teamspace/studios/this_studio/runs/detect'
interesting = ['armA_raw', 'armC_default_640', 'armD_old_320', 'armE_default_320', 'arm1_control']

for name in interesting:
    csv_path = os.path.join(runs_base, name, 'results.csv')
    args_path = os.path.join(runs_base, name, 'args.yaml')
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', '0').strip()))
            print(f"=== {name} ===")
            print(f"  Best epoch: {best.get('epoch','?')}")
            print(f"  mAP50: {best.get('metrics/mAP50(B)','?').strip()}")
            print(f"  Total epochs: {len(rows)}")
    # Check args for model and imgsz
    if os.path.exists(args_path):
        with open(args_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('model:') or line.startswith('imgsz:') or line.startswith('batch:'):
                    print(f"  {line}")
    print()

# Also check the new runs
for name in ['yolo26n_hacked_640', 'yolo11n_hacked_640', 'yolo26n_pure_320', 'yolov8n_pure_320']:
    csv_path = os.path.join('/teamspace/studios/this_studio/runs', name, 'results.csv')
    args_path = os.path.join('/teamspace/studios/this_studio/runs', name, 'args.yaml')
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', '0').strip()))
            print(f"=== {name} ===")
            print(f"  Best epoch: {best.get('epoch','?')}")
            print(f"  mAP50: {best.get('metrics/mAP50(B)','?').strip()}")
            print(f"  Total epochs: {len(rows)}")
    if os.path.exists(args_path):
        with open(args_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('model:') or line.startswith('imgsz:') or line.startswith('batch:') or line.startswith('cls:') or line.startswith('label_smoothing:'):
                    print(f"  {line}")
    print()
