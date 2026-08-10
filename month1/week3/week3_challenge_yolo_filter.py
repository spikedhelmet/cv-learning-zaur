import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('https://ultralytics.com/images/zidane.jpg')
