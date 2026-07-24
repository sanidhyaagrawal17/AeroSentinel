# /data

This directory holds all dataset files for training, validation, and testing.

## Expected Structure

```
data/
├── raw/                    # Original NEU dataset download (1,800 images)
│   └── NEU-DET/
├── processed/              # YOLO-formatted annotations + resized images
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── augmented/              # Generated at training time via Albumentations
└── data.yaml               # YOLO dataset config (paths, class names, nc)
```

## Dataset

- **Source:** NEU Metal Surface Defect Dataset
- **Size:** 1,800 images, 6 classes
- **Classes:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- **Split:** 70/15/15 (train/val/test), stratified by class
- **Format:** YOLO (class x_center y_center w h)

## Notes

- `raw/` is gitignored — download the dataset separately.
- `augmented/` is generated at training time, not committed.
