from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import os

# ==============================
# CROWD CONTROL SYSTEM
# ==============================

MODEL_PATH = "yolo26s.pt"

# Change this to your actual video filename
VIDEO_PATH = "college_crowd.mp4"

# Safe limit
SAFE_LIMIT = 10

# Detection confidence
CONFIDENCE = 0.25

# Load YOLO
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("Check the VIDEO_PATH filename.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

# Output video
output_path = "crowd_detected.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# CSV log
log_file = open(
    "crowd_log.csv",
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(log_file)

csv_writer.writerow([
    "timestamp",
    "frame",
    "person_count",
    "status"
])

frame_number = 0

print("====================================")
print("      CROWD CONTROL SYSTEM")
print("====================================")
print("Video:", VIDEO_PATH)
print("Press Q to stop.")
print("====================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO PERSON DETECTION
    results = model.predict(
        frame,
        imgsz=960,
        conf=CONFIDENCE,
        classes=[0],       # 0 = person
        verbose=False
    )

    result = results[0]

    # Number of detected people
    person_count = len(result.boxes)

    # Crowd status
    if person_count > SAFE_LIMIT:
        status = "OVERCROWDED"
    else:
        status = "NORMAL"

    # Draw YOLO boxes
    annotated_frame = result.plot()

    # Choose display information
    if status == "OVERCROWDED":
        display_color = (0, 0, 255)
    else:
        display_color = (0, 255, 0)

    # Information box
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (400, 125),
        (0, 0, 0),
        -1
    )

    # Person count
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        display_color,
        2
    )

    # Status
    cv2.putText(
        annotated_frame,
        f"Status: {status}",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        display_color,
        2
    )

    # Extra alert
    if status == "OVERCROWDED":

        cv2.putText(
            annotated_frame,
            "WARNING: CROWD LIMIT EXCEEDED!",
            (25, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    # Save frame to output video
    out.write(annotated_frame)

    # Save information to CSV
    csv_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        frame_number,
        person_count,
        status
    ])

    # Show video
    cv2.imshow(
        "Crowd Control System",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEAN UP
# ==============================

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print("\n====================================")
print("PROCESSING COMPLETED")
print("====================================")
print("Output video :", output_path)
print("Log file     : crowd_log.csv")
print("====================================")