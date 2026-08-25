import cv2
from ultralytics import YOLO

model = YOLO("month1/week5/train-2/weights/best.pt")

cap = cv2.VideoCapture("month1/week6/drones.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("month1/week6/tracked_output.mp4", fourcc, 30, (frame_width, frame_height))

track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, cls in zip(track_ids, boxes, classes):
            x1, y1, x2, y2 = box
            print(f"Track #{track_id} | Class: {cls} | Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id not in track_history:
                track_history[track_id] = []
            
            # This needs to happen EVERY frame, so it must be outside the 'if' block!
            track_history[track_id].append((cx, cy))

            # Keep only the last 50 positions (so the trail doesn't get infinitely long)
            if len(track_history[track_id]) > 50:
                track_history[track_id].pop(0)

            points = track_history[track_id]
            for i in range(1, len(points)):
                cv2.line(annotated, points[i - 1], points[i], (0, 255, 0), 2)
    

    cv2.imshow("Tracking", annotated)
    out.write(annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

