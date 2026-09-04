# Phase 3.4 — Full Matching Pipeline

---

## Goal

Combine everything you've built into a single end-to-end script: give it a shelf photo, it runs YOLO detection, crops each product, generates a CLIP embedding, queries Qdrant, and returns the identified product name + confidence score. This is the core engine of the entire project.

---

## Concept: Pipeline Architecture

Right now your logic is spread across three separate scripts:

1. `crop_products.py` — YOLO detect + crop + save to disk
2. `qdrant_upload.py` — CLIP encode + upload to Qdrant
3. `qdrant_match.py` — CLIP encode + query Qdrant

The full pipeline merges steps 1 and 3 into a single flow:

```
Raw shelf image
    → YOLO detects bounding boxes
    → For each box: crop the region from the original image
    → CLIP encodes the crop (in memory, no saving to disk)
    → Qdrant returns the closest product match
    → Output: list of (product_name, confidence, bounding_box)
```

The key architectural change: **crops never touch the disk.** In your `crop_products.py`, you save crops as `.jpg` files and then read them back in the matcher. In a real pipeline, you keep the crop as a numpy array in memory, convert it to PIL, encode it, and query — all in one pass. This is faster and cleaner.

---

## Technical Mechanics

### OpenCV to PIL Conversion

Your YOLO model gives you crops as OpenCV numpy arrays (BGR color order). CLIP expects PIL Images (RGB color order). You need to convert:

```python
from PIL import Image
import cv2

# OpenCV crop (BGR)
crop_bgr = img[y1:y2, x1:x2]

# Convert BGR → RGB, then wrap in PIL Image
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
crop_pil = Image.fromarray(crop_rgb)
```

Why not just use `Image.open()`? Because the crop only exists in memory — there's no file on disk to open.

### Pipeline Return Format

Your pipeline should return structured data, not just print statements. Think about what the FastAPI endpoint will need in Phase 4:

```python
# Each detection becomes a dictionary
{
    "product_name": "lays_paprika",
    "confidence": 0.94,
    "bbox": [120, 45, 280, 310],  # x1, y1, x2, y2
    "status": "matched"  # or "unknown"
}
```

---

## Step-by-Step Task

### 1. Create the Pipeline Script

Create `nutivo/pipeline.py`. This is the main script that ties everything together.

Start with the imports and model loading:

```python
import cv2
import os
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(".env")

# Load both models once at startup
yolo_model = YOLO("nutivo/sku-shelf.pt")
clip_model = SentenceTransformer("clip-ViT-B-32")

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
```

### 2. Build the Detection Function

Write a function that takes an image path and returns a list of cropped regions with their bounding boxes. Reuse the logic from your `crop_products.py`, but instead of saving to disk, keep the crops in memory:

```python
def detect_products(image_path, confidence_threshold=0.6, min_size=20):
    """
    Runs YOLO on a shelf image and returns cropped product regions.

    Returns:
        List of dicts, each with:
            - "crop": PIL Image of the cropped product
            - "bbox": [x1, y1, x2, y2] bounding box coordinates
            - "detection_conf": YOLO's confidence score
    """
    # Your YOLO detection + cropping logic goes here
    # Instead of cv2.imwrite(), convert crop to PIL and append to a list
    pass
```

Use your existing `crop_products.py` as a reference. The bounding box clamping and size filtering logic stays the same.

### 3. Build the Identification Function

Write a function that takes a PIL crop and queries Qdrant:

```python
def identify_product(crop_pil, similarity_threshold=0.85):
    """
    Encodes a crop with CLIP and queries Qdrant for the best match.

    Returns:
        dict with "product_name", "confidence", and "status"
    """
    # 1. Encode the crop with CLIP
    # 2. Query Qdrant
    # 3. Check if the top score is above the threshold
    # 4. Return the result as a dictionary
    pass
```

### 4. Tie It Together

Write the main pipeline function:

```python
def scan_shelf(image_path):
    """
    Full pipeline: detect products on a shelf and identify each one.

    Returns:
        List of identified products with names, scores, and positions.
    """
    detections = detect_products(image_path)
    results = []

    for det in detections:
        match = identify_product(det["crop"])
        match["bbox"] = det["bbox"]
        match["detection_conf"] = det["detection_conf"]
        results.append(match)

    return results
```

### 5. Add a CLI Entry Point

At the bottom of the script, make it runnable from the command line:

```python
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "nutivo/source_images/shelf_01.jpg"

    print(f"Scanning: {image_path}")
    results = scan_shelf(image_path)

    print(f"\nFound {len(results)} products:\n")
    for r in results:
        if r["status"] == "matched":
            print(f"  {r['product_name']:25s} | Confidence: {r['confidence']:.2f}")
        else:
            print(f"  {'UNKNOWN':25s} | Best score: {r['confidence']:.2f}")
```

Then run it:

```bash
python nutivo/pipeline.py nutivo/source_images/shelf_01.jpg
```

---

## Checkpoint Questions

1. Why do we convert BGR → RGB before passing to CLIP? What would happen if we skipped this step?
2. Why is it better to keep crops in memory instead of saving them to disk and reading them back?
3. The `detect_products` function takes a `confidence_threshold` parameter. What happens to precision vs recall as you raise this threshold from 0.6 to 0.9?

---

## Challenge (No Guidance)

**Add Visual Output**

Extend your pipeline to also produce an annotated image:

1. Draw bounding boxes on the original shelf image using `cv2.rectangle`.
2. Color-code them: green for matched products, red for unknown.
3. Put the product name as text above each box using `cv2.putText`.
4. Save the annotated image as `nutivo/output_annotated.jpg`.

This is exactly the kind of visual output you'd show in a demo or on the frontend.

---

## Supplemental Reading

**For interviews:**

- **"What is an ML pipeline?"** — "A pipeline is a sequence of processing stages where the output of one stage becomes the input of the next. In production, each stage should be modular (swappable), stateless (no side effects between runs), and independently testable. Our pipeline has three stages: detection (YOLO), feature extraction (CLIP), and retrieval (Qdrant)."
- **"Why separate detection from identification?"** — "Single-stage approaches (like training YOLO to classify specific products) require retraining whenever the product catalog changes. By separating detection (generic 'find any product') from identification (embedding similarity), we can add new products by simply uploading a new reference image — zero retraining."

**For production context:**

- In a real deployment, model loading is the slowest part (~2-5 seconds). You load models once at server startup, not per request. This is why FastAPI + global model variables works well.
- For high-throughput scenarios, you'd batch CLIP encoding: instead of encoding one crop at a time, collect all crops and call `model.encode([crop1, crop2, ...])` once. CLIP can encode ~100 images per second on a GPU in batch mode.
- Edge deployment (e.g., on a phone or Raspberry Pi) would use ONNX-exported models and a local vector index (FAISS) instead of cloud Qdrant.
