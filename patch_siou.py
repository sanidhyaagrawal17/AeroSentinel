import re
import os

METRICS_PATH = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/utils/metrics.py"
LOSS_PATH = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/ultralytics/utils/loss.py"

def patch_metrics():
    with open(METRICS_PATH, 'r') as f:
        content = f.read()

    # 1. Update signature
    content = re.sub(r'(CIoU:\s*bool\s*=\s*False,)(\s*eps:\s*float\s*=\s*1e-7,)', r'\1\n    SIoU: bool = False,\2', content)

    # 2. Update enclosing box guard
    content = content.replace('if CIoU or DIoU or GIoU:', 'if CIoU or DIoU or GIoU or SIoU:')

    # 3. Update inner guard
    content = content.replace('if CIoU or DIoU:  # Distance or Complete IoU', 'if CIoU or DIoU or SIoU:  # Distance or Complete IoU')

    # 4. Insert SIoU math right before CIoU block
    siou_math = """            if SIoU:
                # Angle cost
                sigma = torch.pow(b2_x1 + b2_x2 - b1_x1 - b1_x2, 2) / 4 + torch.pow(b2_y1 + b2_y2 - b1_y1 - b1_y2, 2) / 4 + eps
                sin_alpha_1 = torch.abs((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2) / torch.sqrt(sigma)
                sin_alpha_2 = torch.abs((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2) / torch.sqrt(sigma)
                threshold = math.sin(math.pi / 4)
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                sin_alpha = torch.clamp(sin_alpha, 0.0, 1.0 - eps)
                angle_cost = torch.cos(torch.asin(sin_alpha) * 2)

                # Distance cost
                rho_x = torch.pow(((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2) / (cw + eps), 2)
                rho_y = torch.pow(((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2) / (ch + eps), 2)
                gamma = 2 - angle_cost
                distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

                # Shape cost
                omiga_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + eps)
                omiga_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + eps)
                shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + torch.pow(1 - torch.exp(-omiga_h), 4)

                return iou - 0.5 * (distance_cost + shape_cost)
            if CIoU:"""
    
    # We only inject if it's not already there
    if "if SIoU:" not in content:
        content = re.sub(r'(\s+)if CIoU:\s*#', r'\1' + siou_math.strip() + r'  #', content, count=1)
    
    with open(METRICS_PATH, 'w') as f:
        f.write(content)
    print("Patched metrics.py successfully.")

def patch_loss():
    with open(LOSS_PATH, 'r') as f:
        content = f.read()

    # Route bbox regression through SIoU instead of CIoU
    content = content.replace("CIoU=True", "SIoU=True")
    
    with open(LOSS_PATH, 'w') as f:
        f.write(content)
    print("Patched loss.py successfully.")

if __name__ == "__main__":
    patch_metrics()
    patch_loss()
