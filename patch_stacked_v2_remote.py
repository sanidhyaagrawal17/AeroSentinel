import shutil
import re

BLOCK_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/modules/block.py"
TASKS_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/tasks.py"
INIT_DST = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/nn/modules/__init__.py"

def patch_architecture():
    # Copy modified block.py and tasks.py over
    shutil.copy("ultralytics_block.py", BLOCK_DST)
    shutil.copy("ultralytics_tasks.py", TASKS_DST)
    print("Replaced block.py and tasks.py with BiFPN+CBAM+RFAConv additions.")

    with open(INIT_DST, "r") as f:
        content = f.read()
    
    # Export in __all__
    if '"C2f_RFAConv"' not in content and "'C2f_RFAConv'" not in content:
        content = re.sub(r'("Concat",\s*)', r'\1"BiFPN_Concat",\n    "CustomCBAM",\n    "C2f_RFAConv",\n    "ChannelAttention",\n    "SpatialAttention",\n    ', content)
        print("Patched __init__.py to export C2f_RFAConv.")
    else:
        print("C2f_RFAConv already exported in __init__.py.")

    # Import from .block
    if 'C2f_RFAConv,' not in content:
        content = re.sub(r'(TorchVision,\s*)', r'\1BiFPN_Concat,\n    CustomCBAM,\n    C2f_RFAConv,\n    ChannelAttention,\n    SpatialAttention,\n    ', content)
        print("Patched __init__.py to import C2f_RFAConv from .block.")
    else:
        print("C2f_RFAConv already imported in __init__.py.")
        
    with open(INIT_DST, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_architecture()
