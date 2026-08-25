import cv2
from ultralytics import YOLO

model = YOLO("month1/week5/train-2/weights/best.pt")

cap = cv2.VideoCapture("month1/week6/drone_vid.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("month1/week6/tracked_output.mp4", fourcc, 30, (frame_width, frame_height))

track_history = {}

zone_x1 = 300
zone_y1 = 100
zone_x2 = 900
zone_y2 = 600

zone_counter = set()

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 1.2
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

#    end_x = pos[0] + txt_size[0][0] + margin
#    end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, (zone_x1, zone_y1), (zone_x2, zone_y2), bg_color, thickness)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    # cv2.rectangle(annotated,)
    cv2.rectangle(annotated, (zone_x1, zone_y1), (zone_x2, zone_y2), (0,0,255), 2)

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, cls in zip(track_ids, boxes, classes):
            x1, y1, x2, y2 = box
            # print(f"Track #{track_id} | Class: {cls} | Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
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
            
            if((cx > zone_x1 and cx < zone_x2) and (cy < zone_y2 and cy > zone_y1)):
                if track_id not in zone_counter:
                    # It's a new intrusion!
                    zone_counter.add(track_id)
                    print(f"intruder: {track_id}")
                cv2.putText(annotated, f"Alert: Drone {track_id} entered zone at frame {frame_count}", (zone_x1, zone_y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)

    
    cv2.putText(annotated, f"Intrusions: {len(zone_counter)}", (zone_x1, zone_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
    cv2.imshow("Tracking", annotated)
    # out.write(annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

