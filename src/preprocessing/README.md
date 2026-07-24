# /src/preprocessing

OpenCV-based image preprocessing pipeline.

## Responsibilities

- Image loading (file, webcam, batch directory)
- Resize to 640×640
- Normalization (0–1 float scaling)
- CLAHE contrast enhancement on grayscale channel (clip limit: 2.0)
- Input validation and format conversion

## Key Design Notes

- CLAHE improves crack and corrosion visibility on metallic surfaces but can create false textures at aggressive settings — tune clip limit carefully.
- Downscale 4K+ images at ingestion; retain originals for report embedding.
