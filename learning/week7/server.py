from fastapi import WebSocketDisconnect
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
import cv2
import asyncio
from learning.week7.cv_pipeline import shared_state, state_lock, pipeline
from learning.week7.database import init_db, get_recent_events
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing DB")
    init_db()

    print("Launching CV pipeline")
    # Starting in bg thread to avoid web server block
    cv_thread = threading.Thread(
        target=pipeline,
        args=("learning/week6/drone_vid.mp4",),
        daemon=True # auto kills thread when server stops
    )
    cv_thread.start()

    yield

    print("Shutting down")

app = FastAPI(title="Drone Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health_check():
    return {"status": "ok"}

@app.get("/api/events")
async def get_events(limit: int = 50):
    events = get_recent_events(limit)

    # Convert sqlite3.Row objects to standard Python dictionaries 
    # so FastAPI can turn them into JSON
    return [dict(event) for event in events]

async def generate_frames():
    last_frame_number = -1
    while True:
        with state_lock:
            frame = shared_state['frame']
            current_frame_number = shared_state['frame_number']
        
        if frame is None or current_frame_number == last_frame_number:
            # If the CV pipeline hasn't started yet, wait a tiny bit and try again
            await asyncio.sleep(0.01)
            continue

        last_frame_number = current_frame_number

        # Compress matrix into JPEG bytestring
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Don't hog the CPU — yield control back to FastAPI for a few milliseconds
        # await asyncio.sleep(0.03) # roughly 30 FPS

@app.get('/video/feed')
async def video_feed():
    """
    The endpoint that the browser connects to. 
    It returns a StreamingResponse that never closes!
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/detections")
async def ws_endpoint(websocket:WebSocket):
    await websocket.accept()
    print("Browser connected to websocket")

    last_frame_number = -1

    try:
        while True:
            with state_lock:
                frame_num = shared_state["frame_number"]
                detections = shared_state["detections"]
                intrusions = shared_state["intrusion_count"]

            # Only send if new frame
            if frame_num != last_frame_number:
                last_frame_number = frame_num

                await websocket.send_json({
                    "frame_number":frame_num,
                    "intrusion_count":intrusions,
                    "detections": detections,
                })
                
            # Wait 10ms before checking for new data
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Browser disconnected from WebSocket")

@app.get("/api/zones")
async def get_zones():
    return

@app.post("/api/zones")
async def post_zones(name,x1,y1,x2,y2):
    return

@app.delete("/api/zones/{zone_name}")
async def post_zones(name):
    return

