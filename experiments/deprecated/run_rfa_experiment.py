import os
import subprocess
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import parse_model

# ---------------------------------------------------------
# RFAConv Definitions (From Liuchen1997/RFAConv)
# ---------------------------------------------------------
from einops import rearrange

class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel, bias=False)
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, 
                      padding=kernel_size//2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )
       
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size, n2=self.kernel_size)
        return self.conv(conv_data)

class Bottleneck_RFAConv(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RFAConv(c1, c_, k[0], 1) if k[0] > 1 else Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, k[1], 1) if k[1] > 1 else Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_RFAConv(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_RFAConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Register the custom module in Ultralytics parsing logic
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
import inspect

modules.C2f_RFAConv = C2f_RFAConv
tasks.C2f_RFAConv = C2f_RFAConv

# Dynamically patch parse_model to recognize C2f_RFAConv
source = inspect.getsource(tasks.parse_model)
source = source.replace("C2f,", "C2f,\n            C2f_RFAConv,")
exec(source, tasks.__dict__)

# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------
import multiprocessing
import time

def run_arm1():
    print("\n" + "="*50)
    print("Starting ARM 1: Control (Standard YOLOv8n)")
    print("="*50)
    start_time = time.time()
    
    model = YOLO("yolov8n.pt")
    
    # Check for Lightning AI environment
    num_gpus = torch.cuda.device_count()
    device = [i for i in range(num_gpus)] if num_gpus > 1 else (0 if num_gpus == 1 else "cpu")
    workers = os.cpu_count()
    
    model.train(
        data="configs/real_neu_det.yaml", # Points to REAL NEU-DET data
        epochs=60,
        patience=0,
        batch=16,
        imgsz=640,
        device=device,
        workers=workers,
        optimizer="AdamW",
        lr0=0.001,
        # Option A configs
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        flipud=0.0,
        freeze=5,
        name="arm1_control"
    )
    
    elapsed = time.time() - start_time
    print(f"ARM 1 completed in {elapsed / 3600:.2f} hours")
    return elapsed

def run_arm2():
    print("\n" + "="*50)
    print("Starting ARM 2: RFAConv (YOLOv8n-RFAConv)")
    print("="*50)
    start_time = time.time()
    
    # Assuming yolov8-rfaconv.yaml is created in the same directory
    model = YOLO("yolov8-rfaconv.yaml")
    # Load pretrained weights into the standard layers if possible
    model.load("yolov8n.pt")
    
    # Check for Lightning AI environment
    num_gpus = torch.cuda.device_count()
    device = [i for i in range(num_gpus)] if num_gpus > 1 else (0 if num_gpus == 1 else "cpu")
    workers = os.cpu_count()
    
    model.train(
        data="configs/real_neu_det.yaml", # Points to REAL NEU-DET data
        epochs=60,
        patience=0,
        batch=16,
        imgsz=640,
        device=device,
        workers=workers,
        optimizer="AdamW",
        lr0=0.001,
        # Option A configs
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        flipud=0.0,
        freeze=5,
        name="arm2_rfaconv"
    )
    
    elapsed = time.time() - start_time
    print(f"ARM 2 completed in {elapsed / 3600:.2f} hours")
    return elapsed

if __name__ == "__main__":
    print("ARM 1 already completed in previous run.")
    t1 = 0
    t2 = run_arm2()
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print(f"ARM 1 Time: previously completed")
    print(f"ARM 2 Time: {t2/3600:.2f} hrs")
    print(f"Total Time for ARM 2: {t2/3600:.2f} hrs")
