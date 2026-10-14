import time
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

model = YOLO("learning/drone-model.pt")
cap = cv2.VideoCapture("learning/drone_vid.mp4")
frame_list = []
DEVICE = "mps"

while cap.isOpened():
    success, frame = cap.read()
    if not success or len(frame_list) >= 300:
        break

    frame_list.append(frame)

cap.release()

times = []

for frame in frame_list:
    t0 = time.perf_counter()
    results = model.predict(frame, device=DEVICE, verbose=False)
    wall_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    assert isinstance(result, Results)   # runtime check, and it narrows the type for Pyrefly

    preprocess = result.speed["preprocess"]
    inference = result.speed["inference"]
    postprocess = result.speed["postprocess"]

    start_plot = time.perf_counter()
    annotated_frame = result.plot()
    end_plot = time.perf_counter()
    plot_time_ms = (end_plot - start_plot) * 1000

    start_enc = time.perf_counter()
    success, buffer = cv2.imencode(".jpg", annotated_frame)
    end_enc = time.perf_counter()
    encode_time_ms = (end_enc - start_enc) * 1000

    times.append({
        "preprocess": preprocess,
        "inference": inference,
        "postprocess": postprocess,
        "plot time": plot_time_ms,
        "encode time": encode_time_ms,
    })

stats = times[10:]                       
keys = ["preprocess", "inference", "postprocess", "plot time", "encode time"]
means = {k: sum(t[k] for t in stats) / len(stats) for k in keys + ["wall clock"]}
total = sum(means.values())

print(f"Device: {DEVICE}")
print(f"Total: {total}")
for k in keys:
    print(f"{k:<12} {means[k]:>8.2f} ms {means[k] / total * 100:>6.1f} %")