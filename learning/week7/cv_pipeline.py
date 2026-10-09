import threading
from learning.week7.database import log_event
from ultralytics import YOLO
import time
import cv2

state_lock = threading.Lock()
shared_state = {
    "frame": None, 
    "detections": [],
    "intrusion_count": 0,
    "frame_number" : 0,
    "is_paused": False,
}

model = YOLO("learning/drone-model.pt")

def pipeline(file_path):
    cap = cv2.VideoCapture(file_path)
    
    zone_x1, zone_y1, zone_x2, zone_y2 = 300, 100, 900, 600
    zone_counter = set()
    track_history = {}
    frame_count = 0

    while cap.isOpened():
        if(shared_state["is_paused"] == True):
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        annotated = results[0].plot()
        
        cv2.rectangle(annotated, (zone_x1, zone_y1), (zone_x2, zone_y2), (0,0,255), 2)

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()

            for track_id, box, cls, conf in zip(track_ids, boxes, classes, confidences):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                is_in_zone = (cx > zone_x1 and cx < zone_x2) and (cy < zone_y2 and cy > zone_y1)

                if track_id not in track_history:
                    track_history[track_id] = []
                    
                track_history[track_id].append((cx, cy))

                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

                points = track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(annotated, points[i - 1], points[i], (0, 255, 0), 2)
                    
                if is_in_zone:
                    cv2.putText(annotated, f"Alert: Drone {track_id} entered zone", (zone_x1, zone_y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
                        
                    if track_id not in zone_counter:
                        zone_counter.add(track_id)
                
                # We log to the database successfully!
                log_event(time.time(), frame_count, track_id, cls, conf, is_in_zone)

        intrusion_count = len(zone_counter)
        cv2.putText(annotated, f"Intrusions: {intrusion_count}", (zone_x1, zone_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)

        with state_lock:
            shared_state["frame"] = annotated.copy()
            shared_state["frame_number"] = frame_count

    cap.release()
    cv2.destroyAllWindows()
