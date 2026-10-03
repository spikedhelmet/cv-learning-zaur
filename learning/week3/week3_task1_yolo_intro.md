# Week 3 — Task 1: What Neural Networks Actually Do

---

## Goal

Understand what a neural network is at a practical level — enough to use YOLO intelligently, debug training issues, and answer interview questions. No calculus, no backpropagation derivations. Just the mental model you need as an engineer who *uses* these tools.

---

## Concept: From Classical CV to Deep Learning

For the last two weeks, you've been writing rules by hand:
- "If the pixel is brighter than 127, mark it white" (thresholding)
- "If the background changed, something moved" (MOG2)
- "If this corner moved 12px to the right, track it" (optical flow)

The fundamental problem: **you had to define every rule yourself.** What threshold? What color range? What contour area is "big enough"?

Deep learning flips this. Instead of writing rules, you show the computer thousands of labeled examples ("this image contains a drone, this one doesn't") and it **learns its own rules** from the data.

### Defense Context
In counter-drone systems, hand-tuned rules break constantly. A drone at 200m altitude looks different from one at 50m. Sunrise vs. sunset changes the color profile. Cloud cover affects contrast. A neural network trained on diverse examples handles all of these automatically — that's why every modern defense detection system uses them.

### Web/TS Analogy
Think of classical CV like writing a complex RegExp to validate email addresses — you keep adding edge cases and it never quite covers everything. Deep learning is like using a trained spam classifier: you don't write rules, you feed it labeled examples and it figures out the patterns. You're going from hand-written `if/else` logic to a system that generates its own decision logic from data.

---

## The Three CV Tasks

Before touching YOLO, understand the three fundamental tasks in computer vision. Every model, paper, and job description refers to these:

| Task | Input | Output | Example |
|------|-------|--------|---------|
| **Classification** | One image | One label | "This image contains a drone" |
| **Object Detection** | One image | Multiple bounding boxes + labels + confidence scores | "There are 2 drones at these coordinates with 94% and 87% confidence" |
| **Segmentation** | One image | Pixel-level mask for each object | "These exact pixels belong to drone #1, these to drone #2" |

**We care about Detection.** That's what YOLO does. Classification is too simple (it doesn't tell you *where* the object is). Segmentation is overkill for most real-time systems (too slow, and a bounding box is good enough for tracking).

---

## What a "Model" Actually Is

A model has two parts:

1. **Architecture** — The structure of the neural network. How many layers, what type of layers, how they're connected. Think of this as a TypeScript interface definition — it describes the *shape* of the computation, but contains no actual data. YOLO's architecture is defined in code by the Ultralytics team.

2. **Weights** — The actual learned numbers. Millions of floating-point values that encode what the model has learned from training data. This is the `.pt` file you'll download (e.g., `yolo11n.pt`). Think of this as the serialized state of a trained model — like a JSON blob that represents all the knowledge.

**Inference** = feeding an image through the architecture using the trained weights to get predictions. This is what you'll do first.

**Training** = showing the model labeled examples and adjusting the weights to reduce errors. This is what you'll do in Week 5.

### Model Size Variants
YOLO comes in multiple sizes. Same architecture, different number of layers/parameters:

| Variant | Params | Speed | Accuracy | Use Case |
|---------|--------|-------|----------|----------|
| Nano (n) | ~3M | Fastest | Lowest | Edge devices, drones, real-time on weak hardware |
| Small (s) | ~11M | Fast | Good | Balanced real-time |
| Medium (m) | ~26M | Medium | Better | Server-side processing |
| Large (l) | ~43M | Slow | High | When accuracy matters more than speed |
| XLarge (x) | ~69M | Slowest | Highest | Offline analysis, benchmarks |

For defense/real-time, you almost always start with Nano or Small.

---

## Technical Mechanics: Installing and Running YOLO

### Install Ultralytics

The `ultralytics` package is the official Python library for YOLO. It includes everything: model loading, inference, training, export.

```bash
pip install ultralytics
```

This installs PyTorch as a dependency (the deep learning framework YOLO runs on). It's a large download (~2GB for PyTorch + dependencies). Be patient.

### The YOLO Inference API

```python
from ultralytics import YOLO

# YOLO(model_path: str) -> YOLO
#   model_path: path to a weights file (.pt) or a model name string.
#   If the file doesn't exist locally, Ultralytics auto-downloads it.
#   'yolo11n.pt' = YOLO v11 Nano (smallest, fastest).
model = YOLO('yolo11n.pt')

# model(source) -> list[Results]
#   source: can be a file path (str), a URL (str), a numpy array, or 0 for webcam.
#   Returns a list of Results objects, one per image.
results = model('path/to/image.jpg')

# For a single image, take the first result:
result = results[0]
```

### The Results Object

```python
# result.boxes -> Boxes object containing all detections
# result.boxes.xyxy -> tensor of shape (N, 4): [x1, y1, x2, y2] for each detection
#   (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
# result.boxes.conf -> tensor of shape (N,): confidence score (0.0 to 1.0) for each detection
# result.boxes.cls  -> tensor of shape (N,): class ID (integer) for each detection

# result.names -> dict mapping class IDs to human-readable names
#   e.g., {0: 'person', 1: 'bicycle', 2: 'car', ...}

# result.plot() -> numpy array (BGR image)
#   Draws all bounding boxes, labels, and confidence scores onto the image.
#   Returns a standard NumPy array you can pass to cv2.imshow().
annotated = result.plot()
```

---

## Step-by-Step Task: Build `week3_yolo_image.py`

Create `month2/week3/week3_yolo_image.py`.

### 1. Load the model
Import `YOLO` from `ultralytics` and `cv2`. Load the `yolo11n.pt` model. The first time you run this, it will download the weights file (~6MB for Nano).

### 2. Run inference on a sample image
Use a URL as the source. The Ultralytics team hosts sample images:
```
https://ultralytics.com/images/zidane.jpg
```
Or use any image file you have locally.

### 3. Inspect the raw results
Before using `result.plot()`, manually print the raw detection data to understand what YOLO actually returns:

```python
result = results[0]

# Print the raw bounding boxes, confidence scores, and class IDs
print("Boxes (xyxy):", result.boxes.xyxy)
print("Confidence:", result.boxes.conf)
print("Class IDs:", result.boxes.cls)
print("Class names:", result.names)
```

Look at what gets printed. Each row in `xyxy` is one detected object. Match the class IDs to the names dict.

### 4. Draw and display
Use `result.plot()` to get the annotated image, then display it with `cv2.imshow`. Use `cv2.waitKey(0)` (not `waitKey(1)`) since this is a still image, not a video loop — you want it to wait indefinitely until you press a key.

### 5. Draw your own boxes (instead of relying on `plot()`)
After getting the auto-annotated view working, write a second version where you manually loop through `result.boxes` and draw the rectangles yourself using `cv2.rectangle` and `cv2.putText`. This proves you understand the data structure and can customize the visualization — which you'll need when building the C2 dashboard.

Hint: `result.boxes.xyxy` returns a PyTorch tensor. To get plain Python numbers, use `.cpu().numpy()` to convert it to a NumPy array, then loop over the rows.

---

## Checkpoint Questions

1. What is the difference between a model's architecture and its weights?
2. If YOLO returns a confidence of `0.45` for a detection, what does that mean in practical terms?
3. Why would you choose YOLO Nano over YOLO XLarge for a drone-mounted camera?
4. What is the difference between Classification, Detection, and Segmentation?

---

## Challenge (No Guidance)

**Multi-image batch inference with filtering:**

Create `week3_challenge_yolo_filter.py`:
1. Download 3-4 different sample images from the internet (people, cars, animals, etc.) and save them locally.
2. Run YOLO inference on all of them in a loop.
3. For each image, filter the results to only show detections with confidence above `0.6`.
4. For each filtered detection, print: the class name, the confidence score (formatted to 2 decimal places), and the bounding box coordinates.
5. Display each annotated image one by one (press any key to advance to the next).

---

## Supplemental Reading

**For interviews:**
- **"What is transfer learning?"** — You'll hear this constantly. The short answer: instead of training a model from scratch (which requires millions of images and weeks of GPU time), you start with a model already trained on a massive general dataset (like COCO, which has 80 object classes) and then fine-tune it on your specific data (drones). The pre-trained weights already "understand" edges, shapes, and textures. You're just teaching it what a drone looks like. We will do this hands-on in Week 5.
- **"What is COCO?"** — Common Objects in Context. A dataset of 330K images with 80 object categories. It's the standard benchmark for object detection. The pre-trained `yolo11n.pt` you're downloading was trained on COCO.

**For production context:**
- The `.pt` file format is PyTorch-specific. In production defense systems, models are exported to **ONNX** (Open Neural Network Exchange) format, which can then be optimized with **TensorRT** for NVIDIA GPUs or **OpenVINO** for Intel hardware. We'll cover ONNX export in Week 5.
- **Prerequisite for this reading:** None. **Takeaway:** Understand that `.pt` is for prototyping, ONNX/TensorRT is for deployment.

**External resources:**
- Ultralytics YOLO docs — Predict mode: https://docs.ultralytics.com/modes/predict/
  - **Prerequisite:** None. **Takeaway:** Reference for all the arguments you can pass to `model()` and all the fields on the `Results` object. Bookmark this — you'll use it constantly.
- 3Blue1Brown — *"But what is a neural network?"* (Chapter 1): https://www.youtube.com/watch?v=aircAruvnKk
  - **Prerequisite:** Basic algebra (no calculus needed). **Takeaway:** Visual intuition for what layers, neurons, and weights represent. Watch at 1.5x speed, ~15 minutes. Directly relevant now.
