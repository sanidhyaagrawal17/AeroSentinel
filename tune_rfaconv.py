import os
import sys
from ultralytics import YOLO

def main():
    print("Starting RFAConv Genetic Algorithm Hyperparameter Tuning (100 Iterations)...", flush=True)
    model = YOLO("yolov8-rfaconv.yaml")
    data_config = "/teamspace/studios/this_studio/configs/real_neu_det.yaml"

    results = model.tune(
        data=data_config,
        epochs=30,
        iterations=100,
        batch=16,
        imgsz=320,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,
        cache='ram',
        workers=8,
        device=0,
        project='aerosentinel_runs',
        name='rfaconv_ga_tune'
    )
    print("RFAConv GA Tuning Completed!", flush=True)

if __name__ == "__main__":
    main()
