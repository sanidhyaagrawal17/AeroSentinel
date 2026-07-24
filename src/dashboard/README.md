# /src/dashboard

Streamlit-based local browser dashboard.

## Responsibilities

- Sidebar: image upload / camera toggle, confidence slider, severity filter
- Main panel: annotated image with bounding boxes + defect labels
- Defect log: timestamped table (class, confidence, severity, bbox coordinates)
- Severity heatmap: spatial overlay aggregating detection locations
- Grad-CAM toggle: on-demand explainability overlay
- PDF export trigger

## Key Design Notes

- Runs entirely locally — no web server or internet access required.
- Use `@st.cache_resource` to pre-warm the ONNX model on startup.
- Cold-start latency of 3–5s on low-spec hardware is acceptable for MRO use case.
- Inference latency counter (<30ms) must be visible in the UI for demo purposes.
