# Week 2 — Task 5: ROI (Region of Interest) Selection & Masking

---

## Goal

Learn how to define and extract a **Region of Interest (ROI)** from a frame, apply masks to isolate specific areas for processing, and use interactive mouse selection to define ROIs at runtime. This is the foundation for every "only process this part of the image" operation in computer vision.

---

## Concept: Why ROI Matters

Every CV pipeline you've built so far processes the **entire frame** — all 640×480 pixels, every single loop iteration. In production, this is wasteful and slow.

**ROI** lets you say: *"Only process this rectangle / polygon / circle of the image."*

### Defense Context
In a counter-drone C2 system:
- The sky is the ROI. You don't need to run detection on the ground, trees, or buildings — only the sky region where drones appear.
- After YOLO detects a target, the bounding box becomes the ROI for the tracker. You run optical flow *only* inside that box instead of the entire frame.
- **Geo-zones:** A restricted airspace boundary drawn on a map translates to a polygon ROI on the camera feed. Anything entering that polygon triggers an alert.

### Web / TS Analogy
Think of ROI like `overflow: hidden` on a `<div>`. The full image exists in memory, but you're only rendering/processing a clipped rectangle of it. Or think of it like Canvas `clip()` — you define a path, and only pixels inside that path get drawn or processed.

---

## Technical Mechanics

### 1. Array Slicing (NumPy ROI) — The Simplest Method
Since frames are just NumPy arrays with shape `(height, width, channels)`, you can extract a rectangle by slicing:

```python
# frame[y_start:y_end, x_start:x_end]
roi = frame[100:300, 200:500]  # 200px tall, 300px wide rectangle
```

**Key detail:** NumPy indexing is `[row, col]` which means `[y, x]` — the opposite of what you'd expect from `(x, y)` coordinates in OpenCV drawing functions.

- `frame[100:300, 200:500]` → Start at row 100, end at row 300. Start at col 200, end at col 500.
- This returns a **view** (not a copy), meaning modifying `roi` modifies the original `frame` directly. Use `roi = frame[100:300, 200:500].copy()` if you need an independent copy.

### 2. Mask-Based ROI — For Non-Rectangular Regions
For circular or polygonal ROIs, create a binary mask (black image with white region where you want to process):

```python
# Create a blank black mask (same size as frame, single channel)
mask = np.zeros(frame.shape[:2], dtype=np.uint8)

# Draw a filled white circle on the mask (center, radius)
cv2.circle(mask, (320, 240), 100, 255, -1)

# Apply mask: only pixels where mask is white survive
masked = cv2.bitwise_and(frame, frame, mask=mask)
```

- `frame.shape[:2]` gives `(height, width)` without the channel dimension — masks are always single-channel (grayscale).
- `cv2.bitwise_and(frame, frame, mask=mask)` keeps pixels where `mask == 255` and sets everything else to black.

### 3. Interactive ROI Selection with `cv2.selectROI`
OpenCV provides a built-in tool for the user to draw a rectangle with the mouse:

```python
# Opens a window where you drag to select a rectangle
# Returns (x, y, w, h) of the selected region
roi_box = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
x, y, w, h = roi_box

# Extract the selected region
roi = frame[y:y+h, x:x+w]
```

- **`fromCenter`** (bool): If `True`, the rectangle expands from the center outward. If `False`, you drag corner-to-corner.
- **`showCrosshair`** (bool): Whether to show crosshair lines while selecting.
- Press **ENTER** or **SPACE** to confirm. Press **C** to cancel.

### 4. Mouse Callback for Custom ROI Drawing
For full control (drawing polygons, freeform regions, etc.), you can register a mouse callback:

```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
    elif event == cv2.EVENT_MOUSEMOVE:
        pass  # Track mouse position
    elif event == cv2.EVENT_LBUTTONUP:
        print(f"Released at ({x}, {y})")

cv2.namedWindow("Window")
cv2.setMouseCallback("Window", mouse_callback)
```

- **`cv2.EVENT_LBUTTONDOWN`**: Left mouse button pressed.
- **`cv2.EVENT_LBUTTONUP`**: Left mouse button released.
- **`cv2.EVENT_MOUSEMOVE`**: Mouse is moving (fires constantly).
- **`flags`**: Contains modifier state (e.g., `cv2.EVENT_FLAG_LBUTTON` is set while the left button is held down).

---

## Step-by-Step Task: Build `week2_roi.py`

Create a new file `month1/week2/week2_roi.py` and implement the following:

1. **Open webcam** and capture the first frame.
2. **Use `cv2.selectROI`** to let the user draw a rectangle on the first frame. Store the `(x, y, w, h)` result.
3. **Enter the main loop.** On each frame:
   - Extract the ROI region using NumPy slicing: `roi = frame[y:y+h, x:x+w]`
   - Convert *only the ROI* to grayscale (not the whole frame).
   - Run Canny edge detection on *only the ROI*.
   - Convert the Canny result back to BGR (`cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)`).
   - Place the processed ROI back into the frame: `frame[y:y+h, x:x+w] = edges_bgr`
   - Draw a rectangle outline around the ROI on the frame using `cv2.rectangle`.
   - Display the frame. The result should show the live camera feed with the selected region replaced by its Canny edges.
4. **Press `q` to quit.** Press `r` to re-select a new ROI (call `cv2.selectROI` again).

**What to observe:**
- Only the selected rectangle should show edges. The rest of the frame stays as a normal camera feed.
- Try selecting different regions — your hand, a textured object, a blank wall — and see how the edge density changes.

---

## Checkpoint Questions
Answer these before writing code:
1. Why is NumPy ROI slicing `frame[y:y+h, x:x+w]` using `[y, x]` order instead of `[x, y]`?
because opencv frames come with y,x instead of usual x,y
2. What happens if you modify a NumPy slice *without* `.copy()`? Does it affect the original array?
yes, it mutates the original
3. When would you use a mask-based ROI instead of a simple rectangle?
for non-rectangular shapes.

---

## Challenge (No Guidance)

**Motion Detection Inside a User-Defined Zone**

Create `week2_challenge_roi_motion.py`:
1. Let the user select an ROI using `cv2.selectROI` on the first frame.
2. Run your background subtraction pipeline (`cv2.createBackgroundSubtractorMOG2`) on *only* the ROI region, not the entire frame.
3. If motion is detected inside the ROI (contours found with area above threshold), draw a red rectangle around the ROI and display `"ALERT: Motion in Zone!"`. If no motion, draw a green rectangle.
4. The rest of the frame outside the ROI should remain untouched (normal camera view).

This simulates a real security/defense use case: defining a restricted zone and alerting when something enters it.

---

## Supplemental Reading

**For interviews:**
- **ROI vs. Masking vs. Cropping:** These terms are often used loosely. Be precise: *cropping* extracts a rectangle and discards the rest. *ROI slicing* gives you a view into the original array (modifications propagate back). *Masking* uses a binary image to selectively zero out pixels without changing the array shape.
- **Why ROI matters for performance:** If your frame is 1920×1080 (2M pixels) but your target is in a 200×200 box (40K pixels), processing only the ROI is **50× faster**. This is critical for real-time systems running on edge hardware.

**For production context:**
- **Tracking + ROI:** Modern trackers (SORT, DeepSORT) use the previous detection's bounding box as the ROI search region for the next frame. This is why tracking is faster than running detection on every frame.
- **Multi-camera C2 systems** define persistent geo-zones (polygon ROIs) on each camera feed. Operators draw these zones on a map, and the system automatically translates them to pixel coordinates on each camera's perspective.

**External resources:**
- OpenCV official tutorial on ROI: https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html (Section: "Image ROI")
- 3Blue1Brown — *"But what is a Convolution?"* (YouTube): While not directly about ROI, this video brilliantly visualizes how pixel operations work on sub-regions of images, which is conceptually what ROI processing does. Useful for building visual intuition.
