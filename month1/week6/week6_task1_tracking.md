# Week 6 — Task 1: Multi-Object Tracking

---

## Goal

Right now, your drone detector treats every frame independently. It sees "a drone" 30 times per second, but it has no idea it's the **same** drone. This week, you will add **tracking** — the ability to assign a persistent ID to each detected object and follow it across frames. By the end of this task, your system will say "Drone #1 is moving northeast at 12 pixels/frame" instead of just "I see a drone."

---

## Concept: Detection vs. Tracking

### The Problem with Detection Alone

Your YOLO model outputs a list of bounding boxes for each frame. Frame 1: `[drone at (100, 200)]`. Frame 2: `[drone at (105, 198)]`. Frame 3: `[drone at (110, 195)]`.

To a human, it's obvious these are the same drone moving right and slightly up. But to the computer, these are three completely unrelated observations. It has zero memory between frames.

This breaks down immediately in real scenarios:
- **Two drones cross paths.** Without tracking, the system can't tell which is which after they cross.
- **A drone disappears behind a tree for 0.5 seconds.** Without tracking, the system thinks the drone vanished and a new one appeared.
- **You need to count how many drones entered an area.** Without tracking, the same drone re-entering gets counted as a new one every time.

### What a Tracker Does

A tracker sits **on top of** your detector. It takes the raw detections from YOLO and:
1. Assigns a unique ID to each new object (Drone #1, Drone #2, etc.).
2. Predicts where each tracked object *should* be in the next frame (using motion models).
3. Matches new detections to existing tracks (using distance/overlap).
4. Handles objects that temporarily disappear (occlusion) and reappear.

The detector answers: **"What is in this frame?"**
The tracker answers: **"Which object is which across time?"**

### Defense Context

In a military C2 (Command and Control) system, tracking is non-negotiable. You don't just need to know "there's a drone." You need:
- **Track ID:** "This is Target #3."
- **Trajectory:** "It entered from the east, heading northwest."
- **Speed:** "Moving at 15 m/s."
- **Time in zone:** "It has been in the restricted area for 47 seconds."
- **Intent classification:** "Based on its flight path, it appears to be conducting surveillance."

None of this is possible without persistent tracking.

---

## Technical Mechanics: How Trackers Work

### SORT (Simple Online and Realtime Tracking)

The most fundamental tracker. It uses two core ideas:

1. **Kalman Filter:** A mathematical model that predicts where an object will be in the next frame based on its current position and velocity. Think of it as the tracker's "guess" before it sees the next frame.

2. **Hungarian Algorithm:** After the Kalman Filter makes its predictions, and YOLO gives new detections, the Hungarian Algorithm finds the optimal way to match predictions to detections. It minimizes the total distance between all predicted positions and all detected positions simultaneously.

**Limitation:** SORT only uses position and box size to match objects. If two objects cross paths, SORT can swap their IDs (called an **ID switch**).

### DeepSORT

Adds a **Re-Identification (ReID) feature extractor** on top of SORT. For each detected object, it extracts a visual "fingerprint" — a 128-dimensional vector that describes what the object looks like. When matching detections to tracks, it considers both position (Kalman Filter) AND appearance (ReID features). This dramatically reduces ID switches.

**Limitation:** The ReID model adds computational cost (slower inference).

### ByteTrack

A smarter approach to handling low-confidence detections. Most trackers throw away detections below a confidence threshold (e.g., 0.5). ByteTrack keeps them and uses them in a second matching pass.

**Why this matters:** When a drone is partially occluded (behind a tree), YOLO might detect it with only 30% confidence. A normal tracker would ignore this. ByteTrack says: "I have an existing track that predicts a drone should be right here, and I have a low-confidence detection right here. They probably match." This makes ByteTrack extremely robust to occlusion.

### BoT-SORT

Combines ByteTrack's low-confidence matching with DeepSORT's appearance features, plus camera motion compensation. Currently one of the best general-purpose trackers.

### Which One Will We Use?

Ultralytics has ByteTrack and BoT-SORT built in. You don't need to install anything extra. We'll start with **ByteTrack** because it's fast and handles occlusion well.

---

## Step-by-Step Task: Add Tracking to Your Drone Detector

### 1. Understand the API

Ultralytics makes tracking almost identical to detection. Instead of `model.predict()`, you use `model.track()`. The key difference is that the results now include a `.id` field for each detection — its persistent track ID.

```python
results = model.track(source=..., show=True)
```

That single change gives you tracking. But to really understand what's happening, you need to build it manually.

### 2. Create `week6_track.py`

Create `month1/week6/week6_track.py`. Start with the basic tracking loop:

```python
import cv2
from ultralytics import YOLO

model = YOLO("month1/week5/train-2/weights/best.pt")

cap = cv2.VideoCapture("month1/week5/dronevid.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use model.track() instead of model.predict()
    # persist=True tells the tracker to remember IDs across frames
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw the results on the frame
    annotated = results[0].plot()

    cv2.imshow("Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run this. You should see bounding boxes with **ID numbers** next to the class name (e.g., "drone #1", "drone #2"). Those IDs should stay consistent as the drone moves across the frame.

**Key argument: `persist=True`**
This tells the tracker to maintain its internal state between calls. Without it, the tracker resets every frame and gives new IDs each time (defeating the entire purpose).

### 3. Extract Track Data Programmatically

Now, instead of just drawing boxes, let's actually extract the track IDs and positions so we can do something with them.

After `results = model.track(...)`, add logic to pull out the data:

```python
if results[0].boxes.id is not None:
    track_ids = results[0].boxes.id.int().cpu().tolist()
    boxes = results[0].boxes.xyxy.cpu().tolist()
    classes = results[0].boxes.cls.int().cpu().tolist()

    for track_id, box, cls in zip(track_ids, boxes, classes):
        x1, y1, x2, y2 = box
        print(f"Track #{track_id} | Class: {cls} | Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
```

**Why `.cpu().tolist()`?** The results come back as PyTorch tensors (which could be on GPU memory). `.cpu()` moves them to regular RAM, and `.tolist()` converts them to plain Python lists/numbers that you can use normally.

### 4. Draw Motion Trails

This is where tracking gets visually impressive. Store each track's history and draw a line showing where it has been.

Create a dictionary outside the main loop to store position history:

```python
track_history = {}  # Dictionary: {track_id: [(x, y), (x, y), ...]}
```

Inside the loop, after extracting track data, calculate each object's center point and append it to the history:

```python
cx = int((x1 + x2) / 2)
cy = int((y1 + y2) / 2)

if track_id not in track_history:
    track_history[track_id] = []
track_history[track_id].append((cx, cy))

# Keep only the last 50 positions (so the trail doesn't get infinitely long)
if len(track_history[track_id]) > 50:
    track_history[track_id].pop(0)
```

Then draw the trail on the frame:

```python
points = track_history[track_id]
for i in range(1, len(points)):
    cv2.line(annotated, points[i - 1], points[i], (0, 255, 0), 2)
```

### 5. Add an FPS Counter

You already know how to do this from Week 3. Add `time.time()` calls before and after the tracking call to measure how fast your pipeline runs.

### 6. Save the Output Video

To save the tracked video with all the annotations, IDs, and trails baked in:

```python
# Before the loop, set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("month1/week6/tracked_output.mp4", fourcc, 30, (frame_width, frame_height))

# Inside the loop, after drawing everything
out.write(annotated)

# After the loop
out.release()
```

You'll need to get `frame_width` and `frame_height` from the capture object before the loop starts:
```python
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

---

## Checkpoint Questions

1. What is the difference between `model.predict()` and `model.track()`? What does `persist=True` do?
2. Why does ByteTrack keep low-confidence detections instead of throwing them away? Give a concrete example of when this helps.
3. What is an "ID switch" in tracking? What causes it?
4. If you wanted to calculate a tracked object's speed in pixels per frame, what information would you need from `track_history`?

---

## Challenge (No Guidance)

**Zone Intrusion Counter**

1. Define a rectangular "restricted zone" on the video frame (draw it as a red rectangle).
2. Track all objects. When a tracked drone's center point enters the zone for the first time, increment a counter and print an alert: `"ALERT: Drone #3 entered restricted zone at frame 450"`.
3. Display the counter on the frame (e.g., "Intrusions: 2").
4. Make sure the same drone entering the zone only triggers the alert once (use the track ID to prevent double-counting).

---

## Supplemental Reading

**For interviews:**
- **"Explain the Kalman Filter in simple terms."** — "It's a two-step predict-update cycle. First, it predicts where an object will be based on its current velocity. Then, when a new measurement (detection) arrives, it blends the prediction with the measurement, weighting each by their uncertainty. Over time, it builds an increasingly accurate model of the object's motion."
- **"What is the Hungarian Algorithm?"** — "It solves the assignment problem: given N predictions and M detections, find the optimal one-to-one matching that minimizes total cost (usually IoU distance or Euclidean distance). It runs in O(n^3) time."
- **"How does ByteTrack differ from SORT?"** — "SORT discards low-confidence detections. ByteTrack does a two-stage matching: first match high-confidence detections to tracks, then match remaining low-confidence detections to unmatched tracks. This recovers occluded objects that other trackers lose."

**For production context:**
- **Visual SLAM + Tracking:** In autonomous drones, tracking is combined with Visual SLAM (Simultaneous Localization and Mapping) to build a 3D map of the environment while simultaneously tracking moving objects within it.
- **Track fusion:** In defense systems, tracks from multiple sensors (visual camera, thermal camera, radar) are fused together. If the visual tracker loses an object behind a building, the radar tracker might still have it, maintaining continuity.
- **Track management:** Production systems have explicit logic for track states: `TENTATIVE` (just appeared, might be noise), `CONFIRMED` (seen for N consecutive frames), `LOST` (not detected for M frames), `DELETED` (lost for too long, remove). ByteTrack handles this internally.

**External resources:**
- ByteTrack paper: https://arxiv.org/abs/2110.06864 — The original paper. Section 3 explains the two-stage association clearly.
- Ultralytics Tracking Docs: https://docs.ultralytics.com/modes/track/ — Official reference for all tracking parameters.
- "Understanding Kalman Filters" — MATLAB video series on YouTube. Excellent visual explanations of prediction and update steps.
