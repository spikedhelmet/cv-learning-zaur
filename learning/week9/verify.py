import cv2
import numpy
from onnx_detector import OnnxDetector
from ultralytics import YOLO

model = YOLO("learning/drone-model.pt")
detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])
cap = cv2.VideoCapture("learning/drone_vid.mp4")
frames = []

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frames.append(frame)
    if len(frames) >= 50:
        break
cap.release()

count_mismatches = 0
cls_mismatches = 0
max_box_diff = 0.0
max_conf_diff = 0.0

for i, f in enumerate(frames):
    onnx_det = detector.detect(f)

    r = model.predict(f, conf=0.25, iou=0.45, rect=False, verbose=False)[0]
    pt_boxes = r.boxes.xyxy.cpu().numpy()          # (k, 4)  x1, y1, x2, y2, original-frame pixels
    pt_conf = r.boxes.conf.cpu().numpy()           # (k,)
    pt_cls = r.boxes.cls.cpu().numpy().astype(int) # (k,)

    # different number of boxes: print both sides so we can look at it
    if len(onnx_det) != len(pt_boxes):
        count_mismatches += 1
        print(f"frame {i}: count differs, onnx={len(onnx_det)} pytorch={len(pt_boxes)}")
        print("  onnx:   ", onnx_det)
        print("  pytorch:", numpy.column_stack([pt_boxes, pt_conf, pt_cls]))
        continue

    # same count: pair each onnx box with the closest pytorch box
    for (x1, y1, x2, y2, conf, cls) in onnx_det:
        box_diffs = numpy.abs(pt_boxes - numpy.array([x1, y1, x2, y2])).max(axis=1)  # (k,)
        j = box_diffs.argmin()

        max_box_diff = max(max_box_diff, box_diffs[j])
        max_conf_diff = max(max_conf_diff, abs(conf - pt_conf[j]))
        if cls != pt_cls[j]:
            cls_mismatches += 1
            print(f"frame {i}: class differs, onnx={cls} pytorch={pt_cls[j]}")

print()
print(f"frames compared:     {len(frames)}")
print(f"count mismatches:    {count_mismatches}")
print(f"class mismatches:    {cls_mismatches}")
print(f"max corner diff:     {max_box_diff:.3f} px")
print(f"max confidence diff: {max_conf_diff:.4f}")
