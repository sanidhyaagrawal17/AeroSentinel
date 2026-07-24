import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.inference.engine import InferenceEngine
from src.explainability.gradcam import ExplainabilityModule
from src.reporting.generator import ReportGenerator

# --- Config & Initialization ---
st.set_page_config(
    page_title="AeroSentinel MRO",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Cache the ONNX inference engine to avoid cold starts on every interaction."""
    return InferenceEngine(), ExplainabilityModule()

engine, explainer = load_models()
reporter = ReportGenerator()

# --- Sidebar ---
st.sidebar.title("AeroSentinel")
st.sidebar.markdown("**AI-Powered Structural Vigilance**")
st.sidebar.markdown("---")

inspector_name = st.sidebar.text_input("Inspector Name", "John Doe")
aircraft_id = st.sidebar.text_input("Aircraft ID", "N12345")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")

# Update engine thresholds from UI
engine.conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, engine.conf_threshold, 0.05)
engine.iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, engine.iou_threshold, 0.05)

severity_filter = st.sidebar.multiselect(
    "Filter by Severity",
    ["Critical", "Medium", "Low"],
    default=["Critical", "Medium", "Low"]
)

show_gradcam = st.sidebar.checkbox("Enable Grad-CAM Overlay", value=False, help="Loads PyTorch model to show explainability heatmap.")

st.sidebar.markdown("---")
st.sidebar.info("System is running fully offline.")

# --- Main App ---
st.title("Edge-Native Defect Detection")

# Image input
upload_col, camera_col = st.columns(2)
with upload_col:
    uploaded_file = st.file_uploader("Upload Fuselage Image", type=["jpg", "jpeg", "png"])
with camera_col:
    # Optional camera input
    use_camera = st.checkbox("Use Device Camera")
    if use_camera:
        camera_img = st.camera_input("Capture Image")
    else:
        camera_img = None

image_to_process = None

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_to_process = cv2.imdecode(file_bytes, 1)
elif camera_img is not None:
    file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
    image_to_process = cv2.imdecode(file_bytes, 1)

if image_to_process is not None:
    if engine.session is None:
        st.error("Model file not found! Please ensure you have trained and exported the ONNX model to the `models` directory.")
    else:
        # Run Inference
        with st.spinner("Processing frame..."):
            results, annotated_img, latency_ms = engine.predict(image_source=image_to_process)
            
            # Filter results
            filtered_results = [r for r in results if r['severity'] in severity_filter]

        # --- Generate Severity Heatmap ---
        def generate_severity_heatmap(detections, image_shape):
            """
            Creates a spatial heatmap where bounding box regions are colored
            by severity: Critical (high intensity) → Medium → Low.
            Returns a BGR image with the heatmap blended onto a neutral background.
            """
            h, w = image_shape[:2]
            heat = np.zeros((h, w), dtype=np.float32)
            
            severity_intensity = {
                "Critical": 255,
                "Medium": 160,
                "Low": 80
            }
            
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                intensity = severity_intensity.get(det['severity'], 80)
                # Additive blending so overlapping defects show higher intensity
                heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], intensity)
            
            # Smooth the heatmap into a continuous gradient
            if heat.max() > 0:
                heat = cv2.GaussianBlur(heat, (99, 99), 0)
            
            # Normalize to 0-255 for colormap
            heat_norm = np.uint8(heat / heat.max() * 255) if heat.max() > 0 else np.uint8(heat)
            heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            
            return heatmap_color
        
        severity_heatmap = generate_severity_heatmap(filtered_results, image_to_process.shape)
        severity_overlay = cv2.addWeighted(image_to_process, 0.6, severity_heatmap, 0.4, 0)
        
        # --- Display Results ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tab_annotated, tab_severity, tab_gradcam = st.tabs([
                "📸 Annotated View", 
                "🔥 Severity Heatmap", 
                "🧠 Grad-CAM"
            ])
            
            with tab_annotated:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Inference Latency: {latency_ms:.1f} ms | Provider: ONNX CPU")
            
            with tab_severity:
                st.image(cv2.cvtColor(severity_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption("Severity heatmap — brighter zones indicate higher severity detections")
            
            with tab_gradcam:
                if show_gradcam:
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        heatmap = explainer.generate_heatmap(image_to_process)
                        gradcam_overlay = cv2.addWeighted(annotated_img, 0.6, heatmap, 0.4, 0)
                        explainer.unload_model()
                    st.image(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption("Grad-CAM attention heatmap — shows regions the model focused on")
                else:
                    st.info("Enable **Grad-CAM Overlay** in the sidebar to generate the explainability heatmap.")
            
        with col2:
            st.subheader("Defect Log")
            
            if len(filtered_results) == 0:
                st.success("No defects detected matching current filters.")
            else:
                # Create a simple dataframe for display
                import pandas as pd
                df = pd.DataFrame([{
                    'Class': r['class_name'], 
                    'Conf': f"{r['confidence']:.2f}",
                    'Severity': r['severity']
                } for r in filtered_results])
                
                # Apply color styling based on severity (heuristic)
                def color_severity(val):
                    color = 'red' if val == 'Critical' else 'orange' if val == 'Medium' else 'green'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df.style.map(color_severity, subset=['Severity']), use_container_width=True, hide_index=True)
                
                # Export Report
                st.markdown("---")
                if st.button("Generate Offline PDF Report", type="primary"):
                    with st.spinner("Generating report..."):
                        report_path = reporter.generate_report(
                            aircraft_id=aircraft_id,
                            inspector_name=inspector_name,
                            results=filtered_results,
                            annotated_image=annotated_img
                        )
                    st.success(f"Report saved to `outputs/{report_path.name}`")
                    
                    # Provide download button
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=report_path.name,
                            mime="application/pdf"
                        )
else:
    # Placeholder when no image is loaded
    st.info("Upload an image or activate the camera to begin inspection.")

