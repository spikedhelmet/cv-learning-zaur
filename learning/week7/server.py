from learning.week7.database import get_recent_events
from learning.week7.database import init_db
from fastapi import WebSocketDisconnect
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
import asyncio
import cv2
from learning.week7.cv_pipeline import shared_state
from learning.week7.cv_pipeline import state_lock
from learning.week7.cv_pipeline import pipeline
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    cv_thread = threading.Thread(target=pipeline, args=("learning/drone_vid.mp4",), daemon=True)
    cv_thread.start()

    yield

app = FastAPI(title="Drone Detection API", lifespan=lifespan)

# Control state
pause_event = asyncio.Event()
pause_event.set()
task_is_running = shared_state["is_paused"]

@app.get("/health")
async def health_check():
    return{"status":"ok", "message":"Api is running"}

@app.get("/api/events")
async def get_events(limit: int=50):
    events = get_recent_events(limit)
    # Turning sqlitre row to json
    return [dict(event) for event in events]

async def generate_frames():
    while True:
        with state_lock:
            frame = shared_state["frame"]
        
        # If pipeline has not started it must wait
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n" b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(0.01)

@app.get('/video/feed')
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    await websocket.accept()
    print("Browser connected to Websocket")

    last_frame_number = 1

    try:
        while True:
            with state_lock:
                frame_num = shared_state["frame_number"]
                detections = shared_state["detections"]
                intrusions = shared_state["intrusion_count"]

            if frame_num != last_frame_number:
                last_frame_number = frame_num

                await websocket.send_json({
                    "frame_number": frame_num,
                    "intrusion_count": intrusions,
                    "detections": detections,
                })
            
            # Wait 10ms to reduce cpu load
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Browser disconnected from WebSocket")

# Pause
@app.post("/api/pause")
async def pause():
    global pause_event
    pause_event.clear()
    shared_state["is_paused"] = True
    return {"status":"paused"}

@app.post("/api/resumed")
async def resume():
    global pause_event
    pause_event.set()
    shared_state["is_paused"] = False
    return {"status":"resumed"}

