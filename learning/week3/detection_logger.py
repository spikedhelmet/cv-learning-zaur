import datetime
from ultralytics import YOLO
import cv2
import time

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)
prev_time = time.time()
unique_classes = set()
prev_frame_cls = set()

while True:
    success, frame = cap.read()
    if not success:
        continue

    current_frame_cls = set()
    results = model(frame, stream=True, verbose=False)
    
    for result in results:
        # curr_time = time.time()
        # fps = 1 / (curr_time - prev_time)
        # prev_time = curr_time
        
        # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # img = result.orig_img
        classes = result.boxes.cls
        boxes_xyxy = result.boxes.xyxy
        conf_numbers = result.boxes.conf.cpu().numpy()        

        for box, cls_id, conf in zip(boxes_xyxy, classes, conf_numbers):
            if conf < 0.6:
                continue
            
            x, y, x2, y2 = box.tolist()
            class_name = result.names[int(cls_id.item())]
            unique_classes.add(class_name)
            current_frame_cls.add(class_name)
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.putText(frame, str(conf), (int(x),int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


    new_objects = current_frame_cls - prev_frame_cls
    lost_objects = prev_frame_cls - current_frame_cls
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for obj in new_objects:
        print(f"[{current_time}] NEW: {obj}")

    for obj in lost_objects:
        print(f"[{current_time}] LOST: {obj}")

    prev_frame_cls = current_frame_cls
    cv2.putText(frame, f"Uniques: {len(unique_classes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()