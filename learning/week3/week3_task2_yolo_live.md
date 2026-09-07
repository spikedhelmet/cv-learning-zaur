# Week 3 — Task 2: Live YOLO Detection with FPS Counter

---

## Goal

Run YOLO object detection on your **live webcam feed** in real-time, measure **Frames Per Second (FPS)**, and understand the performance tradeoffs between model sizes. By the end, you'll have a live detection window with bounding boxes, class labels, confidence scores, and an FPS counter overlay.

---

## Concept: Real-Time Inference

So far, you've run YOLO on static images — load image, detect, display. That's **batch inference**.

**Real-time inference** means running detection inside a `while True:` loop on every frame from your camera, exactly like your motion detector and optical flow scripts. The difference is that instead of classical CV operations (background subtraction, Canny), you're running a neural network on each frame.

The critical question becomes: **How fast can the model process each frame?**

### FPS (Frames Per Second)
FPS measures how many frames your pipeline can process per second.

- **30 FPS** = smooth, real-time video. Each frame is processed in ~33ms.
- **15 FPS** = noticeable lag but still usable.
- **5 FPS** = choppy, significant delay. A fast-moving drone could cross your screen between frames.
- **<1 FPS** = slideshow. Unusable for real-time applications.

### Defense Context
In counter-drone systems, FPS directly impacts response time. If your model runs at 5 FPS, a drone moving at 50 km/h crosses ~2.8 meters between each frame. At 30 FPS, it only moves ~0.46 meters. Faster FPS = more granular tracking = better intercept predictions.

On edge devices (NVIDIA Jetson, drone-mounted cameras), you trade accuracy for speed by using smaller model variants (Nano instead of XLarge).

### Web / TS Analogy
FPS in CV is like `requestAnimationFrame` performance in web development. If your render loop takes 50ms per frame, you're stuck at 20 FPS and animations stutter. You profile with Chrome DevTools to find bottlenecks — here, you'll profile with `time.time()` to find how long inference takes.

---

## Technical Mechanics

### Measuring FPS with `time`
The simplest FPS measurement: record the time before and after processing each frame, then calculate `1 / elapsed_time`.

```python
import time

prev_time = time.time()

while True:
    # ... process frame ...
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

- `time.time()` returns the current time in seconds (as a float).
- `curr_time - prev_time` gives elapsed seconds for one frame.
- `1 / elapsed` converts seconds-per-frame to frames-per-second.
- `f"FPS: {fps:.1f}"` formats the float to 1 decimal place (e.g., `FPS: 24.3`).

### Running YOLO in a Loop
Instead of passing an image path to `model()`, pass each frame directly:

```python
results = model(frame)
```

Ultralytics accepts NumPy arrays (which is what `cap.read()` returns) directly. Each call returns a list of results, but since you're passing a single frame, `results[0]` is the only result.

### The `stream=True` Optimization
When running YOLO in a loop, use `stream=True` to reduce memory usage:

```python
results = model(frame, stream=True)
for result in results:
    # process result
```

Without `stream=True`, YOLO stores all results in memory at once. With `stream=True`, results are generated one at a time (a Python generator). For single-frame inference in a loop, the difference is small, but it's a good habit for when you process video files with hundreds of frames.

### `result.plot()` — The Easy Way
Ultralytics provides a built-in method that draws all boxes, labels, and confidence scores for you:

```python
annotated_frame = result.plot()
cv2.imshow("YOLO", annotated_frame)
```

This is convenient for quick visualization. For your task, you should implement manual drawing (like you did in the challenge) to solidify your understanding, but know that `result.plot()` exists for rapid prototyping.

---

## Step-by-Step Task: Build `week3_yolo_live.py`

Create `month1/week3/week3_yolo_live.py`:

1. **Load the model.** Start with `yolo11n.pt` (Nano — fastest).
2. **Open the webcam** with `cv2.VideoCapture(0)`.
3. **Main loop:**
   - Read frame from camera.
   - Record `start_time = time.time()` before inference.
   - Run `results = model(frame, stream=True)`.
   - For each result, loop through `boxes`, `cls`, and `conf` (like your challenge solution).
   - Filter detections below a confidence threshold (e.g., `0.5`).
   - Draw bounding boxes, class names, and confidence scores on the frame.
   - Calculate FPS from elapsed time and draw it on the frame (top-left corner).
   - Display the frame.
4. **Quit with `q`.**

**What to observe:**
- What FPS do you get with `yolo11n.pt`? Write it down.
- Try switching to `yolo11s.pt` (Small) — does FPS change noticeably?
- Point the camera at different objects. Does YOLO detect your phone? Your keyboard? Your face?
- Wave an object quickly. At what point does the model start missing detections?

---

## Checkpoint Questions
1. Why does FPS drop when you switch from a Nano model to a Small or Medium model?
2. If your pipeline runs at 10 FPS, how many milliseconds does each frame take to process?
3. What is the difference between passing a file path to `model()` vs. passing a NumPy frame?

---

## Challenge (No Guidance)

**Detection Logger**

Create `week3_challenge_detection_logger.py`:
1. Run live YOLO detection on your webcam.
2. Every time a **new class** appears that wasn't in the previous frame (e.g., a `"cell phone"` enters the scene), print a timestamped log message to the terminal: `[2026-08-11 22:45:03] NEW: cell phone (conf: 0.87)`.
3. When a class **disappears** from the frame, print: `[2026-08-11 22:45:05] LOST: cell phone`.
4. Keep a running count of how many unique objects have been seen since the script started, and display it on the frame as `"Total unique classes: 5"`.

Hint: You'll need to compare the set of detected classes in the current frame against the previous frame's set.

---

## Supplemental Reading

**For interviews:**
- **Inference time vs. model size tradeoff:** Interviewers often ask *"How would you optimize a detection pipeline for real-time use?"* The answer starts with model size selection (Nano vs. Large), then moves to hardware acceleration (GPU, TensorRT, ONNX Runtime), then input resolution (smaller input = faster inference but lower accuracy).
- **Batch size in inference:** Processing multiple frames at once (batching) can improve GPU utilization, but adds latency. For real-time single-camera feeds, batch=1 is standard. For offline video processing, larger batches are faster overall.

**For production context:**
- **NVIDIA Jetson benchmarks for YOLO:** The Jetson Orin Nano can run YOLOv8n at ~60 FPS with TensorRT optimization. Without TensorRT, the same model runs at ~15 FPS. Hardware acceleration matters enormously.
- **Frame skipping:** A common production trick — if your model runs at 10 FPS but your camera outputs 30 FPS, you skip 2 out of every 3 frames and only run detection on every 3rd frame. The other frames use the previous detection result. This keeps the video smooth while limiting compute load.

**External resources:**
- Ultralytics YOLO Predict docs: https://docs.ultralytics.com/modes/predict/ — official reference for all prediction parameters (`conf`, `iou`, `stream`, `device`, etc.)
- *"How does YOLO work?"* — Computerphile (YouTube): A solid 15-minute visual explanation of how YOLO divides an image into a grid and predicts boxes per cell. Good for building intuition without drowning in math.
