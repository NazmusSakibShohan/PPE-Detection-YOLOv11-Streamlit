import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import logging

# Set up logging to catch hidden errors
logging.basicConfig(level=logging.WARNING)

# Page Configuration
st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("Smart Construction Safety Monitor")
st.write("Real-time PPE Compliance detection using YOLOv11")

# Load YOLO Model
@st.cache_resource
def load_model():
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

# --- Live Webcam Section ---
elif source_radio == "Live Webcam":
    st.info("Click 'Start' to enable webcam. If the connection hangs, try a different network or browser.")
    
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = 0.25  # Default confidence

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")

                # Perform YOLO Inference
                # We use the threshold from the slider (passed via the processor)
                results = model.predict(img, conf=self.conf, verbose=False)
                
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = img

                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            except Exception as e:
                # Catching errors prevents the 'NoneType' loop crash
                return frame

    # -------------------------------------------------------
    # FIX: Added free TURN servers from Open Relay Project.
    # STUN-only fails on mobile networks / carrier-grade NAT.
    # TURN servers act as a relay when direct P2P is blocked.
    # -------------------------------------------------------
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            # STUN servers (fast, direct path)
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            # TURN servers — relay fallback for mobile/strict NAT
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:80?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    })

    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Dynamically update the confidence threshold in the running processor
    if ctx.video_processor:
        ctx.video_processor.conf = conf_threshold
