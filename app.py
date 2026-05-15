import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import logging

logging.basicConfig(level=logging.WARNING)

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

# ─── TURN Config ──────────────────────────────────────────────────────────────
# Store these in secrets.toml (add to .gitignore):
#
#   TURN_USERNAME   = "b2794eff8ad425615a2f6008"
#   TURN_CREDENTIAL = "dCjTXzhWE6Br/qhR"

turn_username   = st.secrets.get("TURN_USERNAME", "")
turn_credential = st.secrets.get("TURN_CREDENTIAL", "")

TURN_HOST = "shohan.metered.live"  # Your Metered.ca project host

def build_rtc_config(username: str, credential: str) -> RTCConfiguration:
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": [f"stun:{TURN_HOST}:80"]},
    ]

    if username and credential:
        turn_servers = [
            {
                "urls": [f"turn:{TURN_HOST}:80"],
                "username": username,
                "credential": credential,
            },
            {
                "urls": [f"turn:{TURN_HOST}:443"],
                "username": username,
                "credential": credential,
            },
            {
                "urls": [f"turn:{TURN_HOST}:443?transport=tcp"],
                "username": username,
                "credential": credential,
            },
            {
                "urls": [f"turn:{TURN_HOST}:80?transport=tcp"],
                "username": username,
                "credential": credential,
            },
        ]
        ice_servers.extend(turn_servers)
        st.sidebar.success("✅ TURN credentials loaded")
    else:
        st.sidebar.warning(
            "⚠️ TURN credentials missing. "
            "Add TURN_USERNAME and TURN_CREDENTIAL to secrets.toml."
        )

    return RTCConfiguration({"iceServers": ice_servers})


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

# ─── Live Webcam ──────────────────────────────────────────────────────────────
elif source_radio == "Live Webcam":
    st.info(
        "**Before starting the Webcam:**\n"
        "- Ensure ✅ TURN credentials loaded is visible in the Sidebar.\n"
        "- Grant camera permission in your browser.\n"
        "- If the spinner persists for more than 15 seconds, check F12 → Console."
    )

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = 0.25

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                results = model.predict(img, conf=self.conf, verbose=False)
                annotated_frame = results[0].plot() if results else img
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            except Exception:
                return frame

    RTC_CONFIG = build_rtc_config(turn_username, turn_credential)

    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.conf = conf_threshold
