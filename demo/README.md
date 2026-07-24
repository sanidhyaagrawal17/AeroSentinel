# /demo

Demo assets for the 90-second PoC presentation.

## Expected Contents

```
demo/
├── sample_images/         # Known-defect images for demo walkthrough
├── demo_script.md         # Step-by-step demo sequence (see implementation plan)
└── recording/             # Final screen recording output
```

## Demo Script Summary

| # | Action | Duration |
|---|--------|----------|
| 1 | Show network adapter disabled | 5s |
| 2 | Launch Streamlit dashboard | 5s |
| 3 | Upload sample image with known defects | 5s |
| 4 | Show inference (<30ms counter visible) | 10s |
| 5 | Walk through defect log table | 15s |
| 6 | Toggle Grad-CAM overlay | 15s |
| 7 | Show severity heatmap | 10s |
| 8 | Export + open PDF report | 15s |
| 9 | Recap: network still disabled | 10s |

## Notes

- Sample images should have clear, labelled defects for demo reliability.
- Rehearse Step 1 (offline proof) — Windows: Settings → Network; Linux: `ip link show`.
