import cv2
img = cv2.imread('month1/week4/drone_dataset/images/train/cat.jpg')
# img = cv2.imread('./drone_dataset/images/cat.jpg')

img_h, img_w = img.shape[:2]

with open("month1/week4/drone_dataset/labels/train/cat.txt", "r") as f:
    line = f.read().strip()

# print(line)
    
# Split the string by spaces to get the individual numbers
parts = line.split(" ")
class_id = int(parts[0])
x_center = float(parts[1])
y_center = float(parts[2])
box_w = float(parts[3])
box_h = float(parts[4])

center_x_px = x_center * img_w
center_y_px = y_center * img_h
box_width_px = box_w * img_w
box_height_px = box_h * img_h

xmin = int(center_x_px - (box_width_px / 2))
ymin = int(center_y_px - (box_height_px / 2))
xmax = int(center_x_px + (box_width_px / 2))
ymax = int(center_y_px + (box_height_px / 2))

coco = {"category_id": 0, "bbox": [xmin, ymin, box_width_px, box_height_px]}
pascal_voc = {"name": "drone", "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
print("coco", coco)
print("pascal_voc", pascal_voc)
xcenter = (xmin + xmax) / 2 / img_w
ycenter = (ymin + ymax) / 2 / img_h
box_width = (xmax - xmin) / img_w
box_height = (ymax - ymin) / img_h
coco_to_yolo = { 0,  xcenter, ycenter, box_width, box_height}
print("coco to yolo", coco_to_yolo)
# cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
# cv2.imshow("image", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
