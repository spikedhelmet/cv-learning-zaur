from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="C:/Users/user/Desktop/Computer Vision/month1/week4/drone_dataset/data.yaml",
    epochs=10,
    imgsz=320,
    batch=8,
    patience=10,
    verbose=True,
    device="cpu"
)