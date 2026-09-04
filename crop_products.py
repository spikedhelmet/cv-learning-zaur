import os
import cv2
from ultralytics import YOLO

model = YOLO('nutivo/sku-shelf.pt')
source = "shelves"
output_dir = "nutivo/cropped_images"
results = model.predict(source, batch=8) 

for shelf_index,result in enumerate(results, start=1):
    crop_counter = 0
    img = result.orig_img
    img_height, img_width = img.shape[:2]

    boxes_xyxy = result.boxes.xyxy
    conf_numbers = result.boxes.conf.cpu().numpy()
    # good_results = list(filter(lambda x: x > 0.6, conf_numbers))
    

    for box, conf in zip(boxes_xyxy, conf_numbers):
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = box.tolist()
        modded_x1 = max(0, int(x1))
        modded_x2 = min(img_width, int(x2))
        modded_y1 = max(0, int(y1))
        modded_y2 = min(img_height, int(y2))

        if (modded_x2 - modded_x1) < 20 or (modded_y2 - modded_y1) < 20:
            continue

        cropped_img = img[modded_y1:modded_y2, modded_x1:modded_x2]

        # 1. Format the string using f-strings with zero-padding (:02d and :03d)
        filename = f"shelf_{shelf_index:02d}_crop_{crop_counter:03d}.jpg"
        # 2. Combine directory path and filename cleanly
        full_path = os.path.join(output_dir, filename)

        cv2.imwrite(full_path, cropped_img)
        crop_counter += 1