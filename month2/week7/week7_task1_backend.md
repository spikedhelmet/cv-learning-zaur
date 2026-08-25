# Week 7 — Task 1: Defense C2 Dashboard Backend (FastAPI + WebSocket)

---

## Goal

Build a FastAPI backend that runs your YOLO detection + ByteTrack tracking pipeline on a video feed and serves the results to a frontend in real-time via WebSocket. This is the transition from "CV scripts that display an OpenCV window on your machine" to "a networked service that any client (browser, mobile app, another server) can consume."

By the end of this task, you'll have:
1. A FastAPI server that processes video frames through YOLO + ByteTrack.
2. A WebSocket endpoint that streams detection events (JSON) to connected clients.
3. An MJPEG HTTP endpoint that streams the annotated video frames to a browser.
4. A SQLite database logging all detection events with timestamps.

---

## Concept: From Script to Service

Everything you've built so far runs as a standalone Python script:
- You run `python track.py`
- OpenCV opens a window on your monitor
- When you close the window, the program ends

This is fine for development and testing. But in production, the detection system needs to be a **service** — a long-running process that listens for connections and serves data to clients over the network. Think of it like any web API you've built in your day job, except instead of serving database records, you're serving computer vision results.

### Defense Context

In a real counter-drone C2 system, the architecture looks like this:

```
[Camera 1] ──┐
[Camera 2] ──┤──→ [CV Backend Server] ──→ [C2 Dashboard (Browser)]
[Camera 3] ──┘         │                        │
                   [Database]            [Operator Alerts]
```

The CV backend is a service that:
- Ingests video streams from multiple cameras
- Runs detection + tracking on each stream
- Pushes detection events to the frontend in real-time
- Logs everything to a database for post-incident review

You're building a single-camera version of this.

### Web/TS Analogy

If you've used Express.js or NestJS, FastAPI is Python's equivalent. It handles HTTP routes and WebSocket connections. The main difference here is that your "business logic" isn't querying a Postgres database — it's running YOLO inference on video frames.

---

## Technical Mechanics

### 1. FastAPI Basics

FastAPI is a modern Python web framework. If you've used Express, the mental model is nearly identical:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

Run it with: `uvicorn main:app --reload`

FastAPI gives you:
- Automatic OpenAPI docs at `/docs`
- Type validation via Pydantic
- Native WebSocket support
- Async support (important for video processing)

### 2. WebSocket for Real-Time Events

HTTP is request-response: the client asks, the server answers. For real-time detection events, you need a **persistent bidirectional connection** — a WebSocket.

```python
from fastapi import WebSocket

@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Send detection data to the client
        await websocket.send_json({
            "frame": 450,
            "timestamp": "2026-08-25T13:30:00",
            "detections": [
                {"track_id": 3, "class": "drone", "confidence": 0.87, "bbox": [100, 200, 300, 400]},
            ],
            "intrusions": 2
        })
```

The frontend connects once with `new WebSocket("ws://localhost:8000/ws/detections")` and receives a continuous stream of JSON messages — one per frame.

### 3. MJPEG Video Streaming

MJPEG (Motion JPEG) is the simplest way to stream video to a browser. It's just a sequence of JPEG images sent over HTTP with a special `multipart` content type. The browser natively knows how to display this in an `<img>` tag — no JavaScript needed.

```python
from fastapi.responses import StreamingResponse

def generate_frames():
    while True:
        # Get the annotated frame from your CV pipeline
        success, frame = cap.read()
        if not success:
            break
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video/feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
```

In the browser, you just need: `<img src="http://localhost:8000/video/feed" />`

### 4. SQLite for Event Logging

SQLite is a file-based database — no server to install, just a `.db` file. Perfect for logging detection events that you want to query later.

```python
import sqlite3

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        frame INTEGER,
        track_id INTEGER,
        class_name TEXT,
        confidence REAL,
        zone_intrusion BOOLEAN
    )
""")
conn.commit()
```

### 5. Threading: CV Pipeline + Web Server

Here's the architectural challenge: your YOLO pipeline runs a blocking `while True` loop reading frames and running inference. Your FastAPI server also needs to run and handle WebSocket connections. They can't both block the main thread.

The solution is to run the CV pipeline in a **background thread** and have it push results to a shared data structure that the WebSocket endpoint reads from.

```
[Background Thread: CV Pipeline]
    │
    ├── Reads frames from video/camera
    ├── Runs YOLO + ByteTrack
    ├── Writes results to shared_state (dict/queue)
    │
[Main Thread: FastAPI Server]
    │
    ├── WebSocket endpoint reads from shared_state
    ├── MJPEG endpoint reads latest annotated frame from shared_state
    └── REST endpoints query SQLite for historical data
```

Python's `threading` module handles this. The shared state needs to be **thread-safe** — use `threading.Lock()` to prevent the CV thread and web thread from reading/writing simultaneously.

---

## Step-by-Step Task: Build `month2/week7/server.py`

### 1. Install dependencies

```bash
pip install fastapi uvicorn
```

You already have `ultralytics` and `opencv-python` installed.

### 2. Create the project structure

```
month2/week7/
├── server.py          # Main FastAPI application
├── cv_pipeline.py     # YOLO + ByteTrack processing loop (runs in background thread)
├── database.py        # SQLite setup and query functions
└── detections.db      # Created automatically on first run
```

### 3. Build `database.py`

Create a module with functions to:
- `init_db()` — Create the `events` table if it doesn't exist.
- `log_event(timestamp, frame, track_id, class_name, confidence, zone_intrusion)` — Insert a row.
- `get_recent_events(limit=50)` — Query the last N events.

Use `sqlite3` directly — no ORM needed for this.

### 4. Build `cv_pipeline.py`

This is essentially your `track.py` refactored into a class or function that:
- Takes a video source (file path or camera URL) as input.
- Runs the YOLO + ByteTrack loop.
- Instead of calling `cv2.imshow()`, stores the latest annotated frame and detection results in a shared state dictionary.
- Calls `log_event()` from `database.py` when intrusions happen.
- Runs in a background thread (doesn't block the main thread).

The shared state should look something like:
```python
shared_state = {
    "frame": None,          # Latest annotated frame (numpy array)
    "detections": [],       # Current frame's detections
    "intrusion_count": 0,   # Total intrusions
    "frame_number": 0,      # Current frame number
}
```

Protect reads/writes with a `threading.Lock()`.

### 5. Build `server.py`

Create the FastAPI application with these endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Returns `{"status": "ok"}` — basic health check |
| GET | `/video/feed` | MJPEG stream of annotated frames |
| GET | `/api/events` | Returns recent detection events from SQLite (JSON) |
| WebSocket | `/ws/detections` | Streams live detection data as JSON per frame |

On startup, the server should:
1. Initialize the database.
2. Start the CV pipeline in a background thread.
3. Begin serving requests.

### 6. Test it

1. Run the server: `uvicorn server:app --host 0.0.0.0 --port 8000`
2. Open `http://localhost:8000/docs` to see the auto-generated API docs.
3. Open `http://localhost:8000/video/feed` in your browser — you should see the annotated video stream.
4. Open `http://localhost:8000/api/events` — you should see logged detection events as JSON.
5. Test the WebSocket using the browser console:
   ```javascript
   const ws = new WebSocket("ws://localhost:8000/ws/detections");
   ws.onmessage = (event) => console.log(JSON.parse(event.data));
   ```

---

## Checkpoint Questions

1. Why can't you run the CV pipeline and the web server in the same `while True` loop? What problem does threading solve here?
2. What is the difference between an MJPEG stream and a WebSocket video stream? When would you use each?
3. Why do you need a `threading.Lock()` when the CV thread writes to `shared_state` and the web thread reads from it? What could go wrong without it?
4. If you wanted to support 4 camera feeds simultaneously, what would you change in the architecture?

---

## Challenge (No Guidance)

**REST API for Zone Management**

Extend your server with these additional endpoints:

1. `POST /api/zones` — Accept a JSON body with `{"name": "Runway A", "x1": 100, "y1": 200, "x2": 500, "y2": 600}` and add it to an in-memory list of active zones.
2. `GET /api/zones` — Return all active zones as JSON.
3. `DELETE /api/zones/{zone_name}` — Remove a zone by name.
4. Modify your CV pipeline to check tracked objects against **all active zones** (not just a hardcoded rectangle). When a drone enters any zone, the alert should include which zone was breached.

This simulates how a real C2 system works — operators define restricted areas dynamically through the dashboard UI, and the backend enforces them in real-time.

---

## Supplemental Reading

**For interviews:**

- **"Explain the difference between REST and WebSocket."** — REST is stateless request-response (client asks, server answers). WebSocket is a persistent bidirectional connection (both sides can send messages at any time). For CV systems: use REST for historical queries ("show me events from the last hour") and WebSocket for real-time streams ("push every detection to me as it happens").
- **"How would you scale this to handle 50 cameras?"** — The answer involves process-level parallelism (one process per camera or per GPU), message queues (Redis, RabbitMQ) to decouple detection from the API layer, and potentially Kubernetes for orchestration. Know the concept even if you haven't implemented it.

**For production context:**

- **GStreamer pipelines** — In production defense systems, video ingestion is handled by GStreamer (C library), not OpenCV's `VideoCapture`. GStreamer supports hardware-accelerated decoding (NVDEC on NVIDIA GPUs), RTSP with authentication, and complex pipeline routing. OpenCV is used only for the final frame processing.
- **NVIDIA DeepStream** — NVIDIA's end-to-end SDK for video analytics. It handles camera ingestion, inference (TensorRT), tracking, and output in a single optimized pipeline. Defense companies use this for production deployments because it can process 30+ camera streams on a single GPU.

**External resources:**

- FastAPI documentation: https://fastapi.tiangolo.com/
  - **Prerequisite:** Basic understanding of HTTP and REST APIs. **Takeaway:** Reference for route decorators, WebSocket handling, and dependency injection. Focus on the WebSocket tutorial section.
- FastAPI WebSocket tutorial: https://fastapi.tiangolo.com/advanced/websockets/
  - **Prerequisite:** Completing Step 5 of this task. **Takeaway:** How to handle multiple WebSocket clients, connection lifecycle, and error handling.
