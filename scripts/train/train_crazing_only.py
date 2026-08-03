from ultralytics import YOLO
import sys

# Load the heavy YOLO11 Large model for maximum capacity and safer pruning later
model = YOLO('yolo11l.pt')

print("Starting True Crazing Expert Training (YOLO11l) on AWS GPU...")
print("Training on the FULL dataset to maximize Crazing feature extraction!")

# Train for 100 epochs
results = model.train(
    data='split_full.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    project='runs/detect',
    name='crazing_expert_yolo11l'
)

print("Training finished successfully! Model saved to runs/detect/crazing_expert_yolo11l/weights/best.pt")
