import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# ==========================================
# CROWD CONTROL MANAGEMENT SYSTEM
# ==========================================

SAFE_LIMIT = 5
CONFIDENCE = 0.10
MODEL_PATH = "models/yolo26s.pt"

# Page configuration
st.set_page_config(
    page_title="Crowd Control Management System",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Crowd Control Management System")
st.write(
    "YOLO-based person detection, crowd counting, "
    "and overcrowding monitoring."
)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Make sure yolo26s.pt is in the same folder as app.py."
    )
    st.stop()

model = load_model()

# ==========================================
# INPUT SELECTION
# ==========================================

input_type = st.radio(
    "Select Input",
    ["Image", "Video"],
    horizontal=True
)

# ==========================================
# IMAGE DETECTION
# ==========================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload a crowd image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            st.error("Unable to read the image.")
            st.stop()

        # YOLO detection
        results = model.predict(
            frame,
            imgsz=960,
            conf=CONFIDENCE,
            classes=[0],
            verbose=False
        )

        result = results[0]

        # Count people
        person_count = len(result.boxes)

        # Crowd status
        if person_count > SAFE_LIMIT:
            status = "OVERCROWDED"
        else:
            status = "NORMAL"

        # Annotated image
        annotated = result.plot()

        # Display
        st.subheader("Detection Result")

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        # Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "People Count",
                person_count
            )

        with col2:
            st.metric(
                "Safe Limit",
                SAFE_LIMIT
            )

        # Status
        if status == "OVERCROWDED":
            st.error(
                "🚨 ALERT: CROWD LIMIT EXCEEDED!"
            )
        else:
            st.success(
                "✅ NORMAL CROWD LEVEL"
            )


# ==========================================
# VIDEO DETECTION
# ==========================================

else:

    uploaded_video = st.file_uploader(
        "Upload a crowd video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        video_bytes = uploaded_video.read()

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(video_bytes)
        temp_input.close()

        # Open video
        cap = cv2.VideoCapture(
            temp_input.name
        )

        if not cap.isOpened():
            st.error("Unable to open the video.")
            os.remove(temp_input.name)
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Output video
        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        fourcc = cv2.VideoWriter_fourcc(
            *"avc1"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        st.subheader("Processing Video...")

        progress_bar = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_number = 0
        maximum_count = 0

        # Process video
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            results = model.predict(
                frame,
                imgsz=640,
                conf=CONFIDENCE,
                classes=[0],
                verbose=False
            )

            result = results[0]

            person_count = len(result.boxes)

            if person_count > maximum_count:
                maximum_count = person_count

            # Crowd status
            if person_count > SAFE_LIMIT:
                status = "OVERCROWDED"
                text_color = (0, 0, 255)
            else:
                status = "NORMAL"
                text_color = (0, 255, 0)

            # Draw YOLO detections
            annotated = result.plot()

            # Information panel
            cv2.rectangle(
                annotated,
                (10, 10),
                (390, 120),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                f"People Count: {person_count}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2
            )

            cv2.putText(
                annotated,
                f"Status: {status}",
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2
            )

            if status == "OVERCROWDED":

                cv2.putText(
                    annotated,
                    "ALERT: OVERCROWDED!",
                    (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            # Write frame
            out.write(annotated)

            # Update progress
            if total_frames > 0:

                progress = min(
                    frame_number / total_frames,
                    1.0
                )

                progress_bar.progress(progress)

        # Release resources
        cap.release()
        out.release()

        progress_bar.progress(1.0)

        # Remove temporary input
        os.remove(temp_input.name)

        st.success(
            "✅ Video processing completed!"
        )

        # Display statistics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Maximum People Detected",
                maximum_count
            )

        with col2:
            st.metric(
                "Safe Limit",
                SAFE_LIMIT
            )

        if maximum_count > SAFE_LIMIT:

            st.error(
                "🚨 OVERCROWDING DETECTED"
            )

        else:

            st.success(
                "✅ CROWD LEVEL NORMAL"
            )

        # Display processed video
        st.subheader(
            "Processed Video"
        )

        with open(output_path, "rb") as video_file:

            video_bytes = video_file.read()

            st.video(
                video_bytes
            )

        # Clean output file
        os.remove(output_path)

# ==========================================
# PROJECT INFORMATION
# ==========================================

st.sidebar.title("Project Information")

st.sidebar.write(
    """
    **Technology:** YOLO + OpenCV + Streamlit

    **Detection:** Person

    **Safe Limit:** 10 people

    **Input:** Image / Video

    **Output:** Person count and crowd status
    """
)