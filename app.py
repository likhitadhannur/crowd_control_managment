import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np


# ==========================================
# CROWD CONTROL MANAGEMENT SYSTEM
# ==========================================

SAFE_LIMIT = 5
CONFIDENCE = 0.10

# Use your model if it exists in the repository.
MODEL_PATH = "models/yolo26s.pt"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crowd Control Management System",
    page_icon="👥",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("👥 Crowd Control Management System")

st.write(
    "YOLO-based person detection, crowd counting, "
    "and overcrowding monitoring."
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load YOLO model: {e}")
    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read the uploaded image.")
        st.stop()


    # Get image dimensions
    h, w = image.shape[:2]

    # Middle line of the image
    mid_x = w // 2


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        image,
        conf=CONFIDENCE
    )


    # ======================================
    # COUNT PEOPLE
    # ======================================

    left_count = 0
    right_count = 0


    # Get detected boxes
    boxes = results[0].boxes


    if boxes is not None:

        for box in boxes:

            # COCO class 0 = person
            class_id = int(box.cls[0])

            if class_id != 0:
                continue


            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Center of detected person
            person_center_x = (x1 + x2) // 2


            # Count left/right
            if person_center_x < mid_x:
                left_count += 1
            else:
                right_count += 1


    # ======================================
    # TOTAL COUNT
    # ======================================

    total_count = left_count + right_count


    # ======================================
    # ANNOTATED IMAGE
    # ======================================

    annotated = results[0].plot()


    # Draw center line
    cv2.line(
        annotated,
        (mid_x, 0),
        (mid_x, h),
        (255, 255, 0),
        2
    )


    # ======================================
    # CROWD STATUS
    # ======================================

    status = (
        "OVERCROWDED"
        if (
            total_count > SAFE_LIMIT
            or left_count > SAFE_LIMIT
            or right_count > SAFE_LIMIT
        )
        else "NORMAL"
    )


    # ======================================
    # DISPLAY IMAGE
    # ======================================

    st.image(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        ),
        caption="Detection Result"
    )


    # ======================================
    # DISPLAY COUNTS
    # ======================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Left Count",
            left_count
        )

    with col2:
        st.metric(
            "Right Count",
            right_count
        )

    with col3:
        st.metric(
            "Total Count",
            total_count
        )


    # ======================================
    # DISPLAY STATUS
    # ======================================

    if status == "OVERCROWDED":

        st.error(
            "🚨 ALERT: ZONE OVERCROWDED"
        )

    else:

        st.success(
            "✅ NORMAL"
        )
