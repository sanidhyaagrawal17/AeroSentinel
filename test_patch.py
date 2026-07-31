import re

with open("d:\\AeroSentinel TATA AIG 2026\\ultralytics_metrics.py", "r") as f:
    content = f.read()

new_content = re.sub(r'(CIoU:\s*bool\s*=\s*False,)(\s*eps:\s*float\s*=\s*1e-7,)', r'\1\n    SIoU: bool = False,\2', content)

if "SIoU: bool = False" in new_content:
    print("Signature patched successfully.")
else:
    print("Signature patch FAILED.")
