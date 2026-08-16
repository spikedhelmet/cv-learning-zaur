import cv2
img = cv2.imread('/month1/week4/drone_dataset/images/cat.jpg')

img_h, img_w = img.shape[:2]

with open("/month1/week4/drone_dataset/labels/train/cat.txt", "r") as f:
    line = f.read().strip()
    
# Split the string by spaces to get the individual numbers
parts = line.split(" ")
class_id = int(parts[0])
x_center = float(parts[1])
y_center = float(parts[2])
box_w = float(parts[3])
box_h = float(parts[4])
