import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import subprocess
import imageio_ffmpeg

# ==========================================
# CROWD CONTROL MANAGEMENT SYSTEM
# ==========================================

SAFE_LIMIT = 5
CONFIDENCE = 0.10
MODEL_PATH = "models/yolo26s.pt"

# ==========================================
# PAGE CONFIGURATION
# ==========================================

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
        f"Model file '{MODEL_PATH}' was not found."
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
            imgsz=1280,
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

        # Draw detections
        annotated = result.plot()

        st.subheader("Detection Result")

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

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

        # --------------------------------------
        # SAVE INPUT VIDEO
        # --------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        # --------------------------------------
        # OPEN VIDEO
        # --------------------------------------

        cap = cv2.VideoCapture(
            input_file.name
        )

        if not cap.isOpened():
            st.error("Unable to open the video.")
            os.remove(input_file.name)
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

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # --------------------------------------
        # CREATE AVI OUTPUT
        # --------------------------------------

        temp_avi = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi"
        )

        temp_avi.close()

        fourcc = cv2.VideoWriter_fourcc(
            *"XVID"
        )

        out = cv2.VideoWriter(
            temp_avi.name,
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():
            st.error(
                "Unable to create output video."
            )
            cap.release()
            os.remove(input_file.name)
            os.remove(temp_avi.name)
            st.stop()

        # --------------------------------------
        # PROCESS VIDEO
        # --------------------------------------

        st.subheader("Processing Video...")

        progress_bar = st.progress(0)

        frame_number = 0
        maximum_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # YOLO detection
            results = model.predict(
                frame,
                imgsz=640,
                conf=CONFIDENCE,
                classes=[0],
                verbose=False
            )

            result = results[0]

            # Count people
            person_count = len(result.boxes)

            maximum_count = max(
                maximum_count,
                person_count
            )

            # ----------------------------------
            # CROWD STATUS
            # ----------------------------------

            if person_count > SAFE_LIMIT:

                status = "OVERCROWDED"

                text_color = (
                    0,
                    0,
                    255
                )

            else:

                status = "NORMAL"

                text_color = (
                    0,
                    255,
                    0
                )

            # ----------------------------------
            # DRAW YOLO DETECTIONS
            # ----------------------------------

            annotated = result.plot()

            # Information panel
            cv2.rectangle(
                annotated,
                (10, 10),
                (420, 125),
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
                f"Safe Limit: {SAFE_LIMIT}",
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Status: {status}",
                (25, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )

            # Alert
            if status == "OVERCROWDED":

                cv2.putText(
                    annotated,
                    "ALERT: OVERCROWDED!",
                    (25, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            # Write frame
            out.write(annotated)

            # Progress
            if total_frames > 0:

                progress = (
                    frame_number / total_frames
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )

        # --------------------------------------
        # RELEASE VIDEO
        # --------------------------------------

        cap.release()
        out.release()

        progress_bar.progress(1.0)

        # --------------------------------------
        # CONVERT AVI -> H264 MP4
        # --------------------------------------

        output_mp4 = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_mp4.close()

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            temp_avi.name,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_mp4.name
        ]

        result_ffmpeg = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # --------------------------------------
        # CLEAN INPUT
        # --------------------------------------

        os.remove(input_file.name)
        os.remove(temp_avi.name)

        if result_ffmpeg.returncode != 0:

            st.error(
                "Video conversion failed."
            )

            st.code(
                result_ffmpeg.stderr.decode(
                    errors="ignore"
                )
            )

            os.remove(output_mp4.name)
            st.stop()

        # --------------------------------------
        # RESULTS
        # --------------------------------------

        st.success(
            "✅ Video processing completed!"
        )

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

        # --------------------------------------
        # DISPLAY VIDEO
        # --------------------------------------

        st.subheader(
            "🎥 Processed Video"
        )

        with open(
            output_mp4.name,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        st.video(
            video_bytes,
            format="video/mp4"
        )

        # --------------------------------------
        # DOWNLOAD BUTTON
        # --------------------------------------

        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="crowd_detected.mp4",
            mime="video/mp4"
        )

        # Clean output
        os.remove(output_mp4.name)

# ==========================================
# PROJECT INFORMATION
# ==========================================

st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    f"""
    **Technology:** YOLO + OpenCV + Streamlit

    **Detection:** Person

    **Safe Limit:** {SAFE_LIMIT} people

    **Confidence:** {CONFIDENCE}

    **Input:** Image / Video

    **Output:** Person count and crowd status
    """
)