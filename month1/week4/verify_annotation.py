import cv2
img = cv2.imread('month1/week4/drone_dataset/images/train/cat.jpg')
# img = cv2.imread('./drone_dataset/images/cat.jpg')

img_h, img_w = img.shape[:2]

with open("month1/week4/drone_dataset/labels/train/cat.txt", "r") as f:
    line = f.read().strip()

print(line)
    
# Split the string by spaces to get the individual numbers
parts = line.split(" ")
class_id = int(parts[0])
x_center = float(parts[1])
y_center = float(parts[2])
box_w = float(parts[3])
box_h = float(parts[4])


# cv2.rectangle(image, top_left_pt, bottom_right_pt, color_bgr, thickness)
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.rectangle(img, (x_center, y_center), (x_center + box_w, y_center + box_h), (0, 255, 0), 2)
cv2.imshow("image", img)

# You have two issues here:

# Data Type: OpenCV functions like cv2.rectangle draw on pixels. You can't draw a line at pixel 100.5. Coordinates must be integers (int), but right now you are passing floats.
# Normalized vs. Absolute Math: Your x_center value is 0.5714. If you just convert that to an integer, it becomes 0. That's because YOLO stores normalized coordinates (percentages from 0 to 1), not absolute pixel coordinates.
# To draw the box properly, you need to reverse the math you did when you created the annotation:

# Multiply your normalized center, width, and height by the actual image width and height (img_w, img_h) to get the absolute pixel values.
# Calculate the top-left corner (xmin, ymin) and bottom-right corner (xmax, ymax) from those center/width/height pixel values.
# Convert those final corners to int before passing them to cv2.rectangle.