import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import logging
import requests

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

# ─── TURN Credentials from Secrets ───────────────────────────────────────────
# Store in secrets.toml (do not push to git):
#
# [.streamlit/secrets.toml]
# METERED_API_KEY = "your-key"
# METERED_HOST    = "your-id.relay.metered.ca"

metered_api_key = st.secrets.get("METERED_API_KEY", "")
metered_host    = st.secrets.get("METERED_HOST", "")

# ─── RTC Config Builder ───────────────────────────────────────────────────────
def build_rtc_config(api_key: str, host: str) -> RTCConfiguration:
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]

    if api_key and host:
        try:
            url = f"https://{host}/api/v1/turn/credentials?apiKey={api_key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                ice_servers.extend(resp.json())
            else:
                st.sidebar.warning(f"⚠️ TURN API error: {resp.status_code} — check your secrets.")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Unable to reach TURN server: {e}")
    else:
        st.sidebar.info("ℹ️ TURN credentials not found. Webcam may not work on mobile or cloud networks.")

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
        "- Grant camera permission in your browser.\n"
        "- If the spinner persists for more than 15 seconds, check your TURN credentials."
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

    RTC_CONFIG = build_rtc_config(metered_api_key, metered_host)

    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.conf = conf_threshold
