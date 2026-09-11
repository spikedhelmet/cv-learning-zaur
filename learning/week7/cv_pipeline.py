import numpy
from typing import TypedDict
from typing_extensions import Optional, List
from learning.week7.database import log_event
from ultralytics import YOLO
import threading
import time
import cv2

model = YOLO("learning/drone-model.pt")

state_lock = threading.Lock()

class SharedState(TypedDict):
    frame: Optional[numpy.ndarray]
    detections: List
    intrusion_count: int
    frame_number: int
    zone: List

shared_state:SharedState = {
    "frame": None,
    "detections": [],
    "intrusion_count": 0,
    "frame_number": 0,
    "zone": [300, 100, 900, 600], # x1, y1, x2, y2
}

def pipeline(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter("month1/week6/tracked_output.mp4", fourcc, 30, (frame_width, frame_height))

    track_history = {}

    zone_x1,zone_y1,zone_x2,zone_y2 = shared_state["zone"]

    zone_counter = set()

    def __draw_label(img, text, pos, bg_color):
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        color = (0, 0, 0)
        thickness = cv2.FILLED
        margin = 2
        txt_size = cv2.getTextSize(text, font_face, scale, thickness)

        cv2.rectangle(img, (zone_x1, zone_y1), (zone_x2, zone_y2), bg_color, thickness)

    frame_count = 0

    while cap.isOpened():
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

            current_detections = [] 

            for track_id, box, cls, conf in zip(track_ids, boxes, classes, confidences):
                x1, y1, x2, y2 = box
                # print(f"Track #{track_id} | Class: {cls} | Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                is_in_zone = (cx > zone_x1 and cx < zone_x2) and (cy < zone_y2 and cy > zone_y1)

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
                    
                if(is_in_zone):
                    cv2.putText(annotated, f"Alert: Drone {track_id} entered zone at frame {frame_count}", (zone_x1, zone_y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
                        
                    if track_id not in zone_counter:
                        # It's a new intrusion!
                        zone_counter.add(track_id)
                # Log to the db:
                log_event(time.time(), frame_count, track_id, cls, conf, is_in_zone)

                current_detections.append({
                    "track_id": track_id,
                    "class": cls,
                    "confidence": conf,
                    "bbox": box
                })

            intrusion_count = len(zone_counter)

            cv2.putText(annotated, f"Intrusions: {intrusion_count}", (zone_x1, zone_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)

            with state_lock:
                shared_state["frame"] = annotated.copy()
                shared_state["detections"] = current_detections
                shared_state["intrusion_count"] = intrusion_count
                shared_state["frame_number"] = frame_count

    cap.release()
