from dataclasses import dataclass
import logging
import datetime
from ultralytics import YOLO
import cv2
import time

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture("http://192.168.0.120:8080/video")
prev_time = time.time()


@dataclass(frozen=True)
class UniqueObj:
    class_name: str
# unique_classes = set()
prev_frame_cls = set()

alert_cls = ["cell phone","keyboard", "person", "door"]

logging.basicConfig(
    filename='alerts.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

while True:
    success, frame = cap.read()
    if not success:
        continue

    current_frame_cls = set()
    results = model(frame, stream=True, verbose=False)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    for result in results:
        classes = result.boxes.cls
        boxes_xyxy = result.boxes.xyxy
        conf_numbers = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes_xyxy, classes, conf_numbers):
            if conf < 0.6:
                continue
            
            x, y, x2, y2 = box.tolist()
            class_name = result.names[int(cls_id.item())]
            # unique_classes.add(class_name)
            if conf > 0.7:
                current_frame_cls.add(class_name)

            if class_name in alert_cls and conf > 0.7:
                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Alert: {class_name}", (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.putText(frame, str(conf), (int(x),int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                cv2.putText(frame, str(conf), (int(x),int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


    new_objects = current_frame_cls - prev_frame_cls
    # lost_objects = prev_frame_cls - current_frame_cls
    
    for obj in new_objects:
        if obj in alert_cls:
            print(f"[{current_time}] NEW: {obj}")
            logging.info(f"{current_time} - Alert: {obj}")

    # for obj in lost_objects:
    #     print(f"[{current_time}] LOST: {obj}")

    prev_frame_cls = current_frame_cls
    # cv2.putText(frame, f"Uniques: {len(unique_classes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()