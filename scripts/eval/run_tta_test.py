"""
Path A: Test-Time Augmentation (TTA) on the 58.1% anchor model.
No retraining. Compare mAP50 with and without TTA.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    # Locate the 58.1% anchor model
    base = Path(__file__).parent
    candidates = [
        base / "runs" / "detect" / "final_training_run-2" / "weights" / "best.pt",
        base / "runs" / "detect" / "aerosentinel_runs" / "train" / "weights" / "best.pt",
    ]
    
    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break
    
    if model_path is None:
        print("ERROR: Could not find anchor model. Searched:")
        for c in candidates:
            print(f"  - {c}")
        sys.exit(1)
    
    data_config = str(base / "configs" / "real_neu_det.yaml")
    print(f"\n{'='*60}")
    print(f"PATH A: TTA TEST ON 58.1% ANCHOR MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data:  {data_config}")
    
    model = YOLO(str(model_path))
    
    # --- Run WITHOUT TTA (baseline confirmation) ---
    print(f"\n{'─'*60}")
    print("RUN 1: Standard validation (augment=False)")
    print(f"{'─'*60}")
    metrics_no_tta = model.val(data=data_config, augment=False, imgsz=320)
    map50_no_tta = metrics_no_tta.box.map50
    map5095_no_tta = metrics_no_tta.box.map
    
    print(f"\n  mAP50:    {map50_no_tta:.4f} ({map50_no_tta*100:.1f}%)")
    print(f"  mAP50-95: {map5095_no_tta:.4f} ({map5095_no_tta*100:.1f}%)")
    
    # Per-class without TTA
    class_indices_no = metrics_no_tta.box.ap_class_index
    per_class_no = {}
    for idx, c_idx in enumerate(class_indices_no):
        name = model.names[c_idx]
        per_class_no[name] = metrics_no_tta.box.ap50[idx]
    
    # --- Run WITH TTA ---
    print(f"\n{'─'*60}")
    print("RUN 2: Test-Time Augmentation (augment=True)")
    print(f"{'─'*60}")
    metrics_tta = model.val(data=data_config, augment=True, imgsz=320)
    map50_tta = metrics_tta.box.map50
    map5095_tta = metrics_tta.box.map
    
    print(f"\n  mAP50:    {map50_tta:.4f} ({map50_tta*100:.1f}%)")
    print(f"  mAP50-95: {map5095_tta:.4f} ({map5095_tta*100:.1f}%)")
    
    # Per-class with TTA
    class_indices_tta = metrics_tta.box.ap_class_index
    per_class_tta = {}
    for idx, c_idx in enumerate(class_indices_tta):
        name = model.names[c_idx]
        per_class_tta[name] = metrics_tta.box.ap50[idx]
    
    # --- Side-by-side comparison ---
    diff = map50_tta - map50_no_tta
    diff_sign = "+" if diff >= 0 else ""
    
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<22} {'No TTA':>10} {'With TTA':>10} {'Delta':>10}")
    print(f"{'─'*52}")
    print(f"{'Overall mAP50':<22} {map50_no_tta*100:>9.1f}% {map50_tta*100:>9.1f}% {diff_sign}{diff*100:>8.1f}%")
    print(f"{'Overall mAP50-95':<22} {map5095_no_tta*100:>9.1f}% {map5095_tta*100:>9.1f}% {diff_sign}{(map5095_tta-map5095_no_tta)*100:>8.1f}%")
    print(f"{'─'*52}")
    
    all_classes = sorted(set(list(per_class_no.keys()) + list(per_class_tta.keys())))
    for cls in all_classes:
        v_no = per_class_no.get(cls, 0)
        v_tta = per_class_tta.get(cls, 0)
        d = v_tta - v_no
        ds = "+" if d >= 0 else ""
        print(f"{cls:<22} {v_no*100:>9.1f}% {v_tta*100:>9.1f}% {ds}{d*100:>8.1f}%")
    
    print(f"\n{'='*60}")
    if diff > 0.01:
        print(f"✅ TTA HELPS: +{diff*100:.1f}% mAP50. Use augment=True in final submission.")
    elif diff > 0:
        print(f"⚠️  TTA MARGINAL: +{diff*100:.1f}% mAP50. Minimal benefit.")
    else:
        print(f"❌ TTA DOES NOT HELP: {diff_sign}{diff*100:.1f}% mAP50. Skip TTA.")
    print(f"{'='*60}")
