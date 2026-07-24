# /src/reporting

Offline PDF report generator using fpdf2.

## Responsibilities

- Generate structured maintenance report with:
  - Header: aircraft ID, date, inspector name
  - Annotated images with bounding boxes
  - Defect table: class, confidence, severity, coordinates
  - Severity summary
  - Certifying engineer sign-off block
- Export as PDF to `outputs/`

## Key Design Notes

- fpdf2 is low-level — use a rigid template layout, keep it simple and functional.
- Embed original-resolution images in the report (not the 640×640 preprocessed ones).
- All generation is fully offline.
