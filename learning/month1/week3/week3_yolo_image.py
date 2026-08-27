import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('https://ultralytics.com/images/zidane.jpg')

for result in results:
    img = result.orig_img
    boxes_xyxy = result.boxes.xyxy
    cls = result.boxes.cls
    # boxes_numbers = boxes.cpu().numpy()
    for box, cls_id in zip(boxes_xyxy, cls):
        box_numpy = box.cpu().numpy()
        x, y, x2, y2 = box.tolist()
        class_name = result.names[int(cls_id.item())]
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, class_name, (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # annotated_frame = result.plot()
    cv2.imshow("Frame", img)

cv2.waitKey(0)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()