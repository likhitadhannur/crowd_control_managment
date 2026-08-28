from ultralytics import YOLO
import cv2, csv
from datetime import datetime

model = YOLO("yolo26s.pt")
SAFE_LIMIT = 10        # per-zone limit
log_file = open("crowd_log.csv", "a", newline="")
csv_writer = csv.writer(log_file)
if log_file.tell() == 0:
    csv_writer.writerow(["timestamp", "left_count", "right_count", "total", "status"])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    mid_x = w // 2

    results = model.track(frame, persist=True, imgsz=960, conf=0.15, verbose=False)

    left_count = 0
    right_count = 0

    if results[0].boxes.id is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if model.names[int(cls)] == "person":
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2

                if center_x < mid_x:
                    left_count += 1
                else:
                    right_count += 1

    total_count = left_count + right_count
    annotated_frame = results[0].plot()

    cv2.line(annotated_frame, (mid_x, 0), (mid_x, h), (255, 255, 0), 2)

    left_color = (0, 0, 255) if left_count > SAFE_LIMIT else (0, 255, 0)
    right_color = (0, 0, 255) if right_count > SAFE_LIMIT else (0, 255, 0)

    cv2.putText(annotated_frame, f"Left: {left_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, left_color, 3)
    cv2.putText(annotated_frame, f"Right: {right_count}", (mid_x + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, right_color, 3)

    status = "OVERCROWDED" if (left_count > SAFE_LIMIT or right_count > SAFE_LIMIT) else "normal"
    if status == "OVERCROWDED":
        cv2.putText(annotated_frame, "ALERT: ZONE OVERCROWDED!", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          left_count, right_count, total_count, status])
    log_file.flush()

    cv2.imshow("Crowd Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()