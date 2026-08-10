from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('assets/drone.jpg')
result = results[0]
