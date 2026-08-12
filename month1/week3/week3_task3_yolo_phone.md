# Week 3 — Task 3: YOLO on Phone Camera Feed (IP Camera)

---

## Goal

Run your YOLO detection pipeline on a **live video stream from your phone's camera** instead of your laptop webcam. This introduces the concept of IP cameras and network video streams — a core skill for any multi-camera surveillance or C2 system.

---

## Concept: IP Cameras and RTSP/HTTP Streams

So far, you've used `cv2.VideoCapture(0)` which grabs video from the webcam physically connected to your machine. But in real deployments, cameras are rarely plugged directly into the computer running the detection software.

Instead, cameras broadcast their video over the **network** as a URL (like a website, but for video). OpenCV can read these URLs directly:

```python
cap = cv2.VideoCapture("http://192.168.1.42:8080/video")
```

That's it. Same `cap.read()` loop, same `frame` output. The only difference is the source.

### Common Stream Protocols
- **HTTP (MJPEG):** The simplest. The camera serves a continuous stream of JPEG images over HTTP. Easy to set up, works through firewalls, but higher bandwidth usage. This is what phone apps typically use.
- **RTSP (Real-Time Streaming Protocol):** The industry standard for IP security cameras. More efficient (uses H.264/H.265 compression), but harder to set up. URL format: `rtsp://user:pass@192.168.1.42:554/stream`.

### Defense Context
A counter-drone C2 system might have 8-16 cameras spread across a perimeter, each broadcasting an RTSP stream. The central server runs YOLO on all streams simultaneously. Your phone camera simulates one of those remote cameras.

### Web / TS Analogy
This is like fetching data from `http://api.example.com/feed` instead of reading from a local JSON file. The code that processes the data stays the same — you just change the source URL.

---

## Technical Mechanics

### Phone Camera Apps

You'll need an app that turns your phone into an IP camera. The app runs a small web server on your phone, and OpenCV connects to it.

**Android:**
- **IP Webcam** (by Pavel Khlebovich) — Free, reliable. Download from Google Play Store.
  - Open the app → scroll to bottom → tap "Start server"
  - It shows a URL like `http://192.168.1.42:8080`
  - The video stream URL for OpenCV is: `http://192.168.1.42:8080/video`

**iPhone:**
- **DroidCam** or **EpocCam** — Search the App Store.
- Some apps require a companion desktop app. IP Webcam (Android) is the simplest because it's purely network-based.

### Important: Same Wi-Fi Network
Your phone and your computer **must be on the same Wi-Fi network**. The IP address (e.g., `192.168.1.42`) is a local network address. If your phone is on mobile data, your computer can't reach it.

### Testing the Stream
Before writing Python code, open the stream URL in your web browser first:
- Navigate to `http://<phone-ip>:8080` (the app's main page)
- You should see a live preview. If you can see it in the browser, OpenCV can read it too.

### OpenCV with IP Camera

```python
# Replace the IP and port with what your phone app shows
cap = cv2.VideoCapture("http://192.168.1.42:8080/video")

# Everything else is identical to your webcam code!
while True:
    success, frame = cap.read()
    if not success:
        continue
    # ... run YOLO, draw boxes, etc.
```

### Handling Latency
Network streams have inherent latency (delay). The video you see might be 100-500ms behind reality. This is normal for HTTP/MJPEG streams. In production, RTSP with hardware decoding reduces this to ~50ms.

If the stream feels slow or choppy, you can try reducing the resolution in the phone app's settings (e.g., 640×480 instead of 1920×1080). Lower resolution = less data to transfer = faster streaming.

---

## Step-by-Step Task: Build `week3_yolo_phone.py`

1. **Install a camera app** on your phone (IP Webcam for Android is recommended).
2. **Connect your phone and computer to the same Wi-Fi network.**
3. **Start the camera server** on your phone. Note the URL it displays (e.g., `http://192.168.1.42:8080`).
4. **Test the stream** by opening `http://<phone-ip>:8080/video` in your browser. Confirm you see live video.
5. **Create `month1/week3/week3_yolo_phone.py`:**
   - Copy your `week3_yolo_live.py` as a starting point.
   - Change `cv2.VideoCapture(0)` to `cv2.VideoCapture("http://<your-phone-ip>:8080/video")`.
   - Run YOLO on each frame as before. Draw bounding boxes, class names, confidence, and FPS.
   - Add a `cv2.resize(frame, (640, 480))` after reading each frame to normalize the resolution (phone cameras often output at odd resolutions).
6. **Walk around with your phone** and point it at different objects. Observe what YOLO detects from a completely different camera angle than your laptop webcam.

**What to observe:**
- Is the FPS different compared to your laptop webcam? Why might it be?
- How much latency (delay) do you notice between moving the phone and seeing the movement on your screen?
- Does YOLO detect objects differently from the phone's perspective (different angle, different lighting)?

---

## Checkpoint Questions
1. Why must your phone and computer be on the same Wi-Fi network for this to work?
2. What is the practical difference between an HTTP/MJPEG stream and an RTSP stream?
3. If you wanted to process video from 4 phone cameras simultaneously, what would you need to change in your code? (Think about it conceptually — you don't need to implement this.)

---

## Challenge (No Guidance)

**Remote Alert System**

Create `week3_challenge_remote_alert.py`:
1. Connect to your phone's camera stream.
2. Define a list of "alert classes" (e.g., `["person", "cell phone", "knife"]`).
3. If any of the alert classes are detected with confidence > 0.7, draw the bounding box in **red** and display `"ALERT: [class_name]"` in large text at the top of the screen.
4. If no alert classes are detected, draw all boxes in **green** as normal.
5. Log all alerts to a text file (`alerts.log`) with timestamps, so you can review what was detected after the session ends.

This simulates a basic remote surveillance system — a camera in one location, processing in another, with automated alerting.

---

## Supplemental Reading

**For interviews:**
- **Edge vs. Cloud processing:** A common question is *"Where should you run inference — on the camera itself (edge) or on a remote server (cloud)?"* Edge is faster (no network latency) but limited by hardware. Cloud has powerful GPUs but adds network delay. Hybrid approaches (edge for real-time tracking, cloud for re-identification) are common in production.
- **Camera calibration:** When using multiple cameras, each has a different lens distortion and field of view. Calibration computes the intrinsic/extrinsic parameters needed to map pixel coordinates to real-world coordinates. This is critical for multi-camera tracking and triangulation.

**For production context:**
- **Multi-camera C2 architecture:** In defense systems, each camera feed is typically handled by a dedicated processing thread or process. A central orchestrator aggregates detections from all cameras and correlates them (e.g., "the same person was seen on Camera 3 and Camera 7"). This is called **multi-camera multi-target tracking (MCMT)**.
- **Video Management Systems (VMS):** Tools like Milestone XProtect or Nx Witness handle dozens of RTSP streams, recording, playback, and integration with analytics. Understanding how these systems work is valuable for defense-tech roles.

**External resources:**
- IP Webcam app for Android: https://play.google.com/store/apps/details?id=com.pas.webcam — the app you'll use for this task.
- *"IP Cameras and OpenCV"* — PyImageSearch tutorial: A thorough walkthrough of connecting OpenCV to various IP camera types. Covers RTSP, HTTP, and GStreamer pipelines.
