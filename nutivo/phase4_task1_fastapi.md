# Phase 4.1 — FastAPI Backend

---

## Goal

Wrap your pipeline in a FastAPI server with REST endpoints. When you're done, any client (your cousin's app, a web frontend, Postman, curl) can send a shelf photo to your API and get back a JSON list of identified products.

---

## Concept: Why FastAPI?

Right now, your pipeline is a CLI script. To use it, you open a terminal and type `python pipeline.py shelf1.jpeg`. That's fine for you, but your cousin's app can't open a terminal on your laptop.

A REST API turns your pipeline into a **service**: a program that runs continuously, listens for HTTP requests, and returns JSON responses. FastAPI is the standard Python framework for this because:

1. **Async by default** — handles multiple requests concurrently.
2. **Automatic docs** — generates interactive API documentation at `/docs` (Swagger UI).
3. **Type validation** — uses Python type hints to validate request/response data automatically.
4. **Fast** — built on Starlette and Uvicorn, one of the fastest Python web frameworks.

---

## Technical Mechanics

### FastAPI Request Lifecycle

```
Client sends POST /scan with image
    → FastAPI receives the request
    → Saves uploaded file to a temp location
    → Calls your scan_shelf() function
    → Converts the result to JSON
    → Returns HTTP 200 with JSON body
    → Cleans up temp file
```

### Key FastAPI Concepts

| Concept | What it does |
|---------|-------------|
| `@app.post("/scan")` | Decorator that registers a function to handle POST requests to `/scan` |
| `UploadFile` | FastAPI's type for handling file uploads. Has `.read()`, `.filename`, etc. |
| `JSONResponse` | Returns a JSON response with a specific status code |
| `app.on_event("startup")` | Runs code once when the server starts (perfect for loading models) |
| CORS Middleware | Allows your frontend (running on a different port) to call the API |

### Project Structure

```
nutivo/
├── pipeline.py          # Your existing pipeline functions (detect, identify, scan)
├── api/
│   └── server.py        # FastAPI app — imports from pipeline.py
├── embeddings/
│   ├── qdrant_upload.py
│   └── qdrant_match.py
└── sku-shelf.pt
```

The key idea: `server.py` **imports** your pipeline functions. It does NOT duplicate them. Your pipeline logic stays in one place.

---

## Step-by-Step Task

### 1. Install FastAPI

```bash
pip install fastapi uvicorn python-multipart
```

`python-multipart` is needed for file upload handling.

### 2. Refactor pipeline.py for Import

Right now, your `pipeline.py` loads models at the top level (lines 14-21). This means models load immediately when the file is imported. That's fine, but you need to remove the `sys.exit()` call and the `from cv2 import rectangle` import (that was a linter workaround).

Also, the `scan_shelf` function currently saves an annotated image to disk and returns just the results list. For the API, we also need the annotated image returned as bytes. Refactor `scan_shelf` to return both:

```python
def scan_shelf(img_path):
    detections = detect_products(img_path)
    cv2_img = cv2.imread(img_path)
    results = []

    for det in detections:
        # ... your existing matching + drawing logic ...

    # Encode the annotated image to JPEG bytes (in memory, no disk!)
    _, img_bytes = cv2.imencode('.jpg', cv2_img)
    
    return results, img_bytes.tobytes()
```

### 3. Create the FastAPI Server

Create `nutivo/api/server.py`:

```python
import sys
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to the path so we can import pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import scan_shelf

app = FastAPI(title="Nutivo Shelf Scanner API")

# Allow frontend (running on different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan")
async def scan_endpoint(file: UploadFile = File(...)):
    """
    Upload a shelf image. Returns identified products with bounding boxes.
    """
    # 1. Save uploaded file to a temp location
    #    (YOLO needs a file path, it can't read from memory)
    
    # 2. Call your pipeline
    
    # 3. Clean up the temp file
    
    # 4. Return the results as JSON
    pass
```

Fill in the logic for the `/scan` endpoint. Here are the tools you need:

- `tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)` — creates a temp file
- `tmp.write(await file.read())` — writes the uploaded bytes to the temp file
- `os.unlink(tmp_path)` — deletes the temp file after processing

### 4. Add a `/products` Endpoint

Add a GET endpoint that lists all products currently in the Qdrant database:

```python
@app.get("/products")
async def list_products():
    """
    Returns all products in the reference database.
    """
    # Use client.scroll() to get all points from the "products" collection
    # Return a list of product names and their metadata
    pass
```

Qdrant's `client.scroll()` returns all points in a collection. Look up its usage.

### 5. Add a `/scan/annotated` Endpoint

Add a second endpoint that returns the annotated image (the one with bounding boxes drawn on it) instead of JSON:

```python
@app.post("/scan/annotated")
async def scan_annotated_endpoint(file: UploadFile = File(...)):
    """
    Upload a shelf image. Returns the annotated image with bounding boxes.
    """
    # Same as /scan, but return the image bytes with:
    # return Response(content=img_bytes, media_type="image/jpeg")
    pass
```

### 6. Run the Server

```bash
uvicorn nutivo.api.server:app --reload --port 8000
```

Then open `http://localhost:8000/docs` in your browser — FastAPI automatically generates interactive API documentation where you can test your endpoints!

---

## Checkpoint Questions

1. What does `async` do in `async def scan_endpoint`? Why does FastAPI use it?
2. Why do we need to save the uploaded file to disk before passing it to YOLO, instead of passing the bytes directly?
3. What is CORS and why do we need the middleware? What would happen if a frontend on `localhost:3000` tried to call your API on `localhost:8000` without CORS?

---

## Challenge (No Guidance)

**Add a `/products/add` Endpoint**

Create a POST endpoint that accepts:
- An image file (the reference product photo)
- A product name (as a form field or query parameter)
- Optional metadata (health_score, category)

The endpoint should:
1. Encode the image with CLIP.
2. Upload the embedding + metadata to Qdrant.
3. Return a success message with the new product's ID.

This is the endpoint that lets you (or your cousin) add new products to the database without touching the code.

---

## Supplemental Reading

**For interviews:**
- **"Why FastAPI over Flask?"** — "FastAPI is built on ASGI (async), which means it can handle I/O-bound operations concurrently without blocking. Flask is WSGI (synchronous). For an ML API where each request involves model inference (CPU-bound), the difference is less dramatic, but FastAPI's automatic request validation via Pydantic and auto-generated OpenAPI docs make it significantly faster to develop and maintain."
- **"How do you handle model loading in a web server?"** — "Models are loaded once at module import time (or during a startup event) and stored as global variables. Each request reuses the same model instance. This avoids the 2-5 second load time on every request. For multi-worker deployments, each worker process loads its own copy of the model."

**For production context:**
- In production, you'd run Uvicorn behind a reverse proxy like Nginx, with multiple worker processes via Gunicorn: `gunicorn nutivo.api.server:app -w 4 -k uvicorn.workers.UvicornWorker`.
- For very heavy inference, you'd offload model execution to a task queue (Celery + Redis) and return a job ID immediately, letting the client poll for results. This prevents long HTTP timeouts.
- File uploads should be size-limited (e.g., 10MB max) to prevent abuse. FastAPI supports this via middleware.
