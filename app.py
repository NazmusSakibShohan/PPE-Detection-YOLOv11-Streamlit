import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av

# Page Configuration
st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("Smart Construction Safety Monitor")
st.write("Real-time PPE Compliance detection using YOLOv11")

# Load YOLO Model
@st.cache_resource
def load_model():
    # Ensure best.pt is in the same directory as app.py
    return YOLO('best.pt')

model = load_model()

# Sidebar Settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_radio = st.sidebar.radio("Select Source", ["Image Upload", "Live Webcam"])

# --- Image Upload Section ---
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
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)

# --- Live Webcam Section (using WebRTC for Cloud Support) ---
elif source_radio == "Live Webcam":
    st.info("Click 'Start' to enable webcam access. If connection fails, check your network firewall.")
    
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Perform YOLO Inference
            results = model.predict(img, conf=conf_threshold, verbose=False)
            
            # Draw Bounding Boxes
            annotated_frame = results[0].plot()

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Enhanced WebRTC Configuration with multiple STUN servers
    webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]}
            ]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
