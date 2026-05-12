import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("Smart Construction Safety Monitor")
st.write("Real-time PPE Compliance detection using YOLOv8")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_radio = st.sidebar.radio("Select Source", ["Image Upload", "Live Webcam"])

if source_radio == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        results = model.predict(img_array, conf=conf_threshold)
        
        annotated_img = results[0].plot()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

elif source_radio == "Live Webcam":
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        st.info("Webcam is off. Check the box to start.")