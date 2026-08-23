from ultralytics import YOLO

# Point this to YOUR newly trained weights!
model = YOLO("month1/week5/train-2/weights/best.pt")

# Run it on your webcam (source 0) or a downloaded video
# model.predict(source="month1/week5/dronevid.mp4", show=True)
model.export(format="onnx")
