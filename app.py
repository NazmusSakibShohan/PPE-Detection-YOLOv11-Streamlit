import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("🦺 Smart Construction Safety Monitor")
st.write("Real-time PPE Compliance detection using YOLOv11")

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ─── Sidebar Settings ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_radio = st.sidebar.radio("Select Source", ["Image Upload", "Live Webcam"])

# ─── Image Upload ─────────────────────────────────────────────────────────────
if source_radio == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        results = model.predict(img_array, conf=conf_threshold)
        annotated_img = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Detected Image",
                use_container_width=True
            )

# ─── Live Webcam (st.camera_input — no WebRTC/TURN needed) ───────────────────
elif source_radio == "Live Webcam":
    st.info(
        "📸 **Take a Photo → YOLO detection will trigger**\n\n"
        "Click the camera icon → Capture photo → See the results below."
    )

    img_file = st.camera_input("Camera")

    if img_file is not None:
        image = Image.open(img_file)
        img_array = np.array(image)

        with st.spinner("Detecting PPE..."):
            results = model.predict(img_array, conf=conf_threshold)
            annotated_img = results[0].plot()

        # Detection summary
        detections = results[0].boxes
        names = results[0].names

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Detected",
                use_container_width=True
            )

        # Show detection results
        if detections and len(detections) > 0:
            st.success(f"✅ {len(detections)} objects detected")
            st.subheader("Detection Details:")
            for box in detections:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = names[cls_id]
                st.write(f"- **{label}** — confidence: `{conf:.2f}`")
        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold and try again.")
