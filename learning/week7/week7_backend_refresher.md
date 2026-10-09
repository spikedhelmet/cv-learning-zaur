# Week 7 — Backend Refresher (Concurrency & FastAPI)

---

## Goal & Concept

**The Objective:** Transform a locally-running OpenCV tracking script into a production-style networked API that serves video and JSON data over the internet.

**The Concept (Concurrency):** 
A web server (like FastAPI) runs an infinite loop listening for HTTP requests. Your OpenCV pipeline (`while cap.isOpened():`) is also an infinite loop processing heavy matrix math. 
If you put them in the same Python file and run them, one will block the other. The video will freeze while waiting for the web server, or the web server will timeout because OpenCV is hogging the CPU.

To solve this, we use **Threading**. We run the web server on the Main Thread, and we spawn a Background Worker Thread solely dedicated to running the OpenCV loop. 

To let them communicate, we create a global "whiteboard" (`shared_state` dictionary) in the middle. The CV thread writes to it, and the Web thread reads from it. To prevent them from crashing into each other, we use a `threading.Lock()`.

---

## Technical Mechanics & API Overview

- **`threading.Thread(target=..., daemon=True)`:** Spawns a new worker thread. `daemon=True` ensures the background thread automatically dies when you stop the FastAPI server.
- **`threading.Lock()`:** A mutual exclusion lock. When one thread enters a `with state_lock:` block, the other thread must wait outside until it's finished. This prevents "race conditions" where both threads try to edit the same memory at the same microsecond.
- **`@asynccontextmanager` (Lifespan):** FastAPI's way of managing startup and shutdown. Everything before `yield` runs when the server boots. `yield` pauses the function to run the web server. Everything after `yield` runs when you press `Ctrl+C`.
- **`cv2.imencode('.jpg', frame)`:** Converts a raw NumPy matrix into compressed JPEG bytes, which browsers require to render images.

---

## Step-by-Step Task

### Step 1: Set up Shared State (`cv_pipeline.py`)
At the very top of `cv_pipeline.py`, you need a global dictionary that both threads can access, and a lock to protect it.

1. Import `threading`.
2. Define a global `state_lock = threading.Lock()`.
3. Define a global `shared_state` dictionary containing: `frame` (None), `detections` (empty list), `intrusion_count` (0), and `frame_number` (0).

### Step 2: Write to Shared State (`cv_pipeline.py`)
Remove `cv2.imshow("Drone Tracker", annotated)` and `cv2.waitKey(1)` from the bottom of your `while` loop. You no longer want local GUI windows popping up.

Instead, at the bottom of the loop, use the lock to safely update the global dictionary:
```python
with state_lock:
    shared_state["frame"] = annotated.copy()  # .copy() is critical to avoid memory tearing
    shared_state["frame_number"] = frame_count
    # (also update detections and intrusion_count)
```

### Step 3: Launch the Thread (`server.py`)
In `server.py`, set up the FastAPI lifespan manager to launch the CV pipeline when the server boots.

```python
from contextlib import asynccontextmanager
import threading
from learning.week7.cv_pipeline import pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize the Database here
    # 2. Launch the CV Thread here
    cv_thread = threading.Thread(target=pipeline, args=("your_video.mp4",), daemon=True)
    cv_thread.start()
    
    yield # Server runs here
```
Initialize your FastAPI app with `app = FastAPI(lifespan=lifespan)`.

### Step 4: The MJPEG Stream (`server.py`)
Browsers can't display raw OpenCV matrices. They need JPEG bytes. 
Create an async generator function `generate_frames()` that:
1. Loops infinitely (`while True:`).
2. Acquires the `state_lock`, grabs `shared_state["frame"]`, and immediately releases the lock.
3. Uses `cv2.imencode('.jpg', frame)` to compress it.
4. Yields the bytes in HTTP multipart format (separated by a `--frame` boundary).
5. Uses `await asyncio.sleep(0.01)` to pace itself.

Create an endpoint `@app.get("/video/feed")` that returns a `StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")`.

### Step 5: The WebSocket Endpoint (`server.py`)

**Why WebSockets?**
HTTP is a "pull" protocol — the browser has to ask for data every time it wants it. If we used HTTP for bounding boxes, the browser would have to spam the server with requests 30 times a second. 
WebSockets are a "push" protocol. You open the connection once, and the server can shove JSON data down the pipe to the browser instantly, whenever it wants.

**The Thread Locking Rule (Crucial!)**
Imagine `shared_state` is a whiteboard in a locked room. 
- The CV Pipeline takes the key, runs in, quickly draws the newest frame (takes 1 millisecond), and leaves.
- The Web Server takes the key, runs in, reads the data, and sends it over the internet to the browser (takes 50 milliseconds).

If the Web Server holds the key *while* it dictates the data over the internet (`await websocket.send_json`), the CV Pipeline is locked out of the room for 50 milliseconds! The video will stutter and freeze.
**Solution:** The Web Server must enter the room (`with state_lock:`), copy the data to a notepad instantly, LEAVE the room (un-indent), and *then* take 50 milliseconds to send the notepad data over the internet.

**The Code Implementation:**
Add this to `server.py` (make sure to `from fastapi import WebSocket, WebSocketDisconnect` at the top!):

```python
@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    # 1. Accept the connection from the browser
    await websocket.accept()
    print("Browser connected to WebSocket!")
    
    last_frame_number = -1
    
    try:
        while True:
            # 2. Enter the locked room and quickly copy the data
            with state_lock:
                frame_num = shared_state["frame_number"]
                detections = shared_state["detections"]
                intrusions = shared_state["intrusion_count"]
                
            # 3. WE LEFT THE LOCKED ROOM! (Un-indented)
            # Now we check if the frame is new, and send it over the slow internet.
            if frame_num != last_frame_number:
                last_frame_number = frame_num
                
                # Push the data to the browser as a JSON string
                await websocket.send_json({
                    "frame_number": frame_num,
                    "intrusion_count": intrusions,
                    "detections": detections
                })
                
            # Wait 10ms before checking for new data so we don't melt the CPU
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("Browser disconnected from WebSocket.")
```

---

## Checkpoint Questions

1. If you forget to use `annotated.copy()` when writing to `shared_state`, what visual bugs will the web browser experience and why?
2. Why must `await websocket.send_json()` be placed outside the `with state_lock:` indentation block?
3. What is the difference between a standard `return {"data": 1}` API endpoint and a `StreamingResponse` generator?

---

## Challenge (No Guidance)

**Pause/Resume API**
Add two standard REST endpoints to `server.py`: `POST /api/pause` and `POST /api/resume`. 
Modify `cv_pipeline.py` to respect a new boolean flag in `shared_state` called `is_paused`. If paused, the CV pipeline should gracefully sleep without processing frames, and resume exactly where it left off when unpaused.

---

## Supplemental Reading

- **Python GIL (Global Interpreter Lock):** Understand why Python `threading` doesn't actually run on multiple CPU cores simultaneously, but is perfectly fine for I/O bounds and C-extensions (like OpenCV/NumPy which release the GIL).
- **Concurrency vs Parallelism:** Read about why `asyncio` is used for the web server (Concurrency) and `threading` is used for the CV pipeline (Parallelism-ish).
