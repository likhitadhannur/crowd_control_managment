import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

SAFE_LIMIT = 10

@st.cache_resource
def load_model():
    return YOLO("yolo26s.pt")

model = load_model()

st.title("Crowd Control Management System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]
    mid_x = w // 2

    results = model(frame, conf=0.15)

    left_count = 0
    right_count = 0

    for box, cls in zip(
        results[0].boxes.xyxy,
        results[0].boxes.cls
    ):
        if model.names[int(cls)] == "person":
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2

            if center_x < mid_x:
                left_count += 1
            else:
                right_count += 1

    total_count = left_count + right_count

    annotated = results[0].plot()

    cv2.line(
        annotated,
        (mid_x, 0),
        (mid_x, h),
        (255, 255, 0),
        2
    )

status = (
    "OVERCROWDED"
    if total_count > SAFE_LIMIT
    or left_count > SAFE_LIMIT
    or right_count > SAFE_LIMIT
    else "NORMAL"
)

st.image(
    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
    caption="Detection Result"
)

st.metric("Left Count", left_count)
st.metric("Right Count", right_count)
st.metric("Total Count", total_count)

    if status == "OVERCROWDED":
        st.error("ALERT: ZONE OVERCROWDED")
    else:
        st.success("NORMAL")
