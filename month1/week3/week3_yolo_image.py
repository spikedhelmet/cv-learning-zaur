import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('https://ultralytics.com/images/zidane.jpg')

for result in results:
    img = result.orig_img
    boxes_xyxy = result.boxes.xyxy
    # boxes_numbers = boxes.cpu().numpy()
    for box in boxes_xyxy:
        box_numpy = box.cpu().numpy()
        x, y, x2, y2 = box.tolist()
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, )

    # annotated_frame = result.plot()
    cv2.imshow("Frame", img)

cv2.waitKey(0)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()