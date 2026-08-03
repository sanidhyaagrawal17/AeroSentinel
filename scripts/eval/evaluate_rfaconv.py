import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Ensure rfaconv_module is importable and register the custom module
sys.path.append(str(Path(__file__).parent))
import rfaconv_module
import ultralytics.nn.modules as nn_modules

# Monkey-patch ultralytics to recognize C2f_RFAConv
setattr(nn_modules, 'C2f_RFAConv', rfaconv_module.C2f_RFAConv)

if __name__ == '__main__':
    model_path = Path(__file__).parent / "models" / "rfaconv_final_best.pt"
    data_config = str(Path(__file__).parent / "configs" / "real_neu_det.yaml")
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}. Train the model first.")
        sys.exit(1)
        
    print(f"\n--- Loading trained model from {model_path} ---")
    model = YOLO(str(model_path))

    print("\n--- Running evaluation on NEU-DET valid/images split ---")
    metrics = model.val(data=data_config)

    # Extract overall metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS")
    print("="*50)
    print(f"Overall mAP50:     {map50:.4f} ({map50*100:.1f}%)")
    print(f"Overall mAP50-95:  {map50_95:.4f} ({map50_95*100:.1f}%)")
    print("-" * 50)
    
    print("Per-class Metrics (mAP50):")
    class_indices = metrics.box.ap_class_index
    for idx, c_idx in enumerate(class_indices):
        c_name = model.names[c_idx]
        c_map50 = metrics.box.ap50[idx]
        print(f" - {c_name:18s}: {c_map50:.4f}")

    print("\n" + "="*50)
    print("COMPARISON AGAINST BASELINE (ANCHOR)")
    print("="*50)
    anchor_map50 = 0.581
    diff = map50 - anchor_map50
    diff_sign = "+" if diff >= 0 else ""
    
    print(f"Baseline (Anchor) mAP50:  {anchor_map50*100:.1f}%")
    print(f"RFAConv (Iteration 64):   {map50*100:.1f}%")
    print(f"Difference:               {diff_sign}{diff*100:.1f}%")
    print("="*50)
    
    # Save a markdown report
    report_path = Path(__file__).parent / "rfaconv_final_report.md"
    with open(report_path, "w") as f:
        f.write("# RFAConv Final Evaluation Report\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- **mAP50**: {map50:.4f} ({map50*100:.1f}%)\n")
        f.write(f"- **mAP50-95**: {map50_95:.4f} ({map50_95*100:.1f}%)\n\n")
        f.write("## Comparison\n")
        f.write(f"- **Baseline (Anchor) mAP50**: {anchor_map50*100:.1f}%\n")
        f.write(f"- **RFAConv (Tuned) mAP50**: {map50*100:.1f}%\n")
        f.write(f"- **Difference**: {diff_sign}{diff*100:.1f}%\n\n")
        f.write("## Per-class Breakdown (mAP50)\n")
        for idx, c_idx in enumerate(class_indices):
            c_name = model.names[c_idx]
            c_map50 = metrics.box.ap50[idx]
            f.write(f"- **{c_name}**: {c_map50:.4f}\n")
    
    print(f"\nReport saved to {report_path}")
