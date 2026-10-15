from ultralytics import YOLO
model = YOLO("learning/drone-model.pt")
path = model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
print(path)