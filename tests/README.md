# /tests

End-to-end and unit tests.

## Expected Contents

| File | Description |
|------|-------------|
| `test_preprocessing.py` | Validate resize, normalization, CLAHE output shapes |
| `test_inference.py` | Verify ONNX model loads, outputs correct bbox format |
| `test_severity.py` | Validate severity triage rules against known inputs |
| `test_e2e.py` | Feed 50 unseen images → verify full pipeline output |
| `test_offline.py` | Confirm zero network calls during entire pipeline |
| `test_report.py` | Validate PDF generation, correct content embedding |

## Notes

- E2E test should run with network adapter disabled.
- Test images stored in `demo/sample_images/`.
