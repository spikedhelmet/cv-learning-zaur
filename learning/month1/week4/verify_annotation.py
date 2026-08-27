import cv2
img = cv2.imread('month1/week4/drone_dataset/train/images/0008_jpg.rf.ed07cab8d3fa5695da33ef0203326a19.jpg')
# img = cv2.imread('./drone_dataset/images/cat.jpg')

img_h, img_w = img.shape[:2]

with open("month1/week4/drone_dataset/train/labels/0008_jpg.rf.ed07cab8d3fa5695da33ef0203326a19.txt", "r") as f:
    line = f.read().strip()

print(line)
    
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


# cv2.rectangle(image, top_left_pt, bottom_right_pt, color_bgr, thickness)
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
