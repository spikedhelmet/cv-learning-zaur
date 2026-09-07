import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
source = "assets/imgs"
results = model.predict(source, batch=8) 

# for i, result in enumerate(results):
#     print(f"--- Image {i} Results ---")

for result in results:
    img = result.orig_img
    classes = result.boxes.cls
    boxes_xyxy = result.boxes.xyxy

    conf_numbers = result.boxes.conf.cpu().numpy()
    # good_results = list(filter(lambda x: x > 0.6, conf_numbers))

    for box, cls_id, conf in zip(boxes_xyxy, classes, conf_numbers):
        if conf < 0.6:
            continue

        x, y, x2, y2 = box.tolist()
        class_name = result.names[int(cls_id.item())]
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, class_name, (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(img, str(conf), (int(x),int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # annotated_frame = result.plot()
    cv2.imshow("Frame", img)
    cv2.waitKey(0)


