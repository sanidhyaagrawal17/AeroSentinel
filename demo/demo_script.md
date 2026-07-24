## Demo Script (90 seconds)

For the virtual PoC presentation, follow this exact sequence:

| # | Action | What judges see | Duration |
|---|--------|-----------------|----------|
| 1 | Show network adapter disabled | **Windows:** Settings → Network → show "No internet". **Linux:** run `ip link show` in terminal showing interfaces DOWN. Screenshot/terminal must be visible. | 5s |
| 2 | Launch Streamlit dashboard | Clean UI loads locally | 5s |
| 3 | Upload sample fuselage image with known defects | Image appears in main panel | 5s |
| 4 | Inference runs | Bounding boxes + severity labels drawn, <30ms counter visible | 10s |
| 5 | Walk through defect log table | Timestamped entries with class, confidence, severity | 15s |
| 6 | Toggle Grad-CAM overlay | Heatmap shows what the model "sees" | 15s |
| 7 | Show severity heatmap | Aggregate structural risk zones highlighted | 10s |
| 8 | Export PDF report → open it | Full structured report with annotated images | 15s |
| 9 | Recap: network still disabled | Reinforce zero-connectivity claim | 10s |
