import shutil
import re

BLOCK_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/modules/block.py"
TASKS_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/tasks.py"
INIT_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/modules/__init__.py"
METRICS_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/utils/metrics.py"
LOSS_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/utils/loss.py"

def patch_architecture():
    # Copy modified block.py and tasks.py over
    shutil.copy("ultralytics_block.py", BLOCK_DST)
    shutil.copy("ultralytics_tasks.py", TASKS_DST)
    print("Replaced block.py and tasks.py with BiFPN additions.")

    # Patch __init__.py to export BiFPN_Concat
    with open(INIT_DST, "r") as f:
        content = f.read()
    
    # Export in __all__
    if '"BiFPN_Concat"' not in content and "'BiFPN_Concat'" not in content:
        content = re.sub(r'("Concat",\s*)', r'\1"BiFPN_Concat",\n    ', content)
        print("Patched __init__.py to export BiFPN_Concat.")
    else:
        print("BiFPN_Concat already exported in __init__.py.")

    # Import from .block
    if 'BiFPN_Concat,' not in content:
        content = re.sub(r'(TorchVision,\s*)', r'\1BiFPN_Concat,\n    ', content)
        print("Patched __init__.py to import BiFPN_Concat from .block.")
        
    with open(INIT_DST, "w") as f:
        f.write(content)

def patch_metrics():
    with open(METRICS_DST, "r") as f:
        content = f.read()

    # Update signature for WIoU
    if "WIoU: bool = False" not in content:
        content = re.sub(r'(CIoU:\s*bool\s*=\s*False,)', r'\1\n    WIoU: bool = False,', content)

    # Update enclosing box guard
    content = content.replace('if CIoU or DIoU or GIoU:', 'if CIoU or DIoU or GIoU or WIoU:')

    # WIoU v3 math string
    wiou_math = """            if WIoU:
                if not hasattr(bbox_iou, 'iou_mean'):
                    bbox_iou.iou_mean = 1.0

                d_center = torch.pow((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2, 2) + \\
                           torch.pow((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2, 2)
                d_enclosing = cw.pow(2) + ch.pow(2) + eps
                
                R_wiou = torch.exp(d_center / d_enclosing)
                L_wiou_v1 = R_wiou * (1 - iou)
                
                iou_val = (1 - iou).detach().mean().item()
                bbox_iou.iou_mean = 0.99 * bbox_iou.iou_mean + 0.01 * iou_val
                
                beta = (1 - iou).detach() / (bbox_iou.iou_mean + eps)
                alpha, delta = 1.9, 3.0
                r = beta / (delta * torch.pow(alpha, beta - delta) + eps)
                
                return 1.0 - (r * L_wiou_v1)
            if CIoU:"""

    if "if WIoU:" not in content:
        content = re.sub(r'(\s+)if CIoU:\s*#', r'\1' + wiou_math.strip() + r'  #', content, count=1)
        with open(METRICS_DST, "w") as f:
            f.write(content)
        print("Patched metrics.py successfully with WIoU v3.")
    else:
        # Check if the buggy import torch exists and remove it
        if "import torch" in content and "hasattr(bbox_iou" in content:
            content = content.replace("import torch\n                if not hasattr(bbox_iou", "if not hasattr(bbox_iou")
            with open(METRICS_DST, "w") as f:
                f.write(content)
            print("Removed buggy import torch from metrics.py.")
        else:
            print("metrics.py already patched.")

def patch_loss():
    with open(LOSS_DST, "r") as f:
        content = f.read()
    
    if "WIoU=True" not in content:
        content = content.replace("CIoU=True", "WIoU=True")
        content = content.replace("SIoU=True", "WIoU=True")
        with open(LOSS_DST, "w") as f:
            f.write(content)
        print("Patched loss.py successfully to route WIoU.")
    else:
        print("loss.py already patched.")

if __name__ == "__main__":
    patch_architecture()
    # patch_metrics() # Disabled for True BiFPN run (keeping default CIoU)
    # patch_loss()    # Disabled for True BiFPN run (keeping default CIoU)
