import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train11/weights/best_fixed.pt")

video_path = "traffic6.mov"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Could not read video.")
    exit()

frame_height = frame.shape[0]
line_y = frame_height // 2
frame_width = frame.shape[1]


counted_ids = set()
vehicle_counts = {name: 0 for name in model.names.values()}
track_ages = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run ByteTrack through YOLO's built-in interface
    results = model.track(source=frame, persist=True, stream=True, tracker="bytetrack.yaml")

    for result in results:
        for box in result.boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])
            track_ages[track_id] = track_ages.get(track_id, 0) + 1
            class_id = int(box.cls[0])
            class_name = model.model.names[class_id]

            # Draw YOLO box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # Count if crossing the line
            if track_id not in counted_ids and track_ages[track_id] > 10:
                counted_ids.add(track_id)
                vehicle_counts[class_name] += 1

            label = f"{class_name} ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


    # Draw counts
    y_offset = 30
    for cls, count in vehicle_counts.items():
        cv2.putText(frame, f"{cls}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    display = cv2.resize(frame, (1280, 720))
    cv2.imshow("ByteTrack Vehicle Count", display)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
