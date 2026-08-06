# Week 2 — Task 4: Sparse Optical Flow (Lucas-Kanade)

---

## Goal & Concept

**Objective:** Build a real-time feature tracking script using Lucas-Kanade Optical Flow.

**What Optical Flow is:** So far, Background Subtraction told you *"something moved in this region."* Optical Flow tells you *"this specific point moved 12 pixels right and 5 pixels up."* It calculates a velocity vector `(dx, dy)` for tracked points between consecutive frames.

**Defense Context:** In counter-drone C2 systems, Background Subtraction detects a target entering the frame. Optical Flow estimates the target's trajectory and speed vector, allowing the system to predict where the target will be 1 second from now.

**Web/TS Analogy:** This is identical to tracking drag velocity in web animations. On every `onPointerMove` event, you compute `deltaX = currentX - prevX` and `deltaY = currentY - prevY` to determine drag direction and speed. Optical Flow does this automatically for pixel features between video frames.

### Sparse vs Dense

There are two flavors:

- **Sparse (Lucas-Kanade):** Tracks a small set of "interesting" points (corners, high-contrast features). Fast, runs at 200+ FPS on embedded hardware. Only tracks specific points, not whole objects.
- **Dense (Farneback):** Calculates motion for every single pixel. Complete motion field, but computationally expensive.

We focus on Sparse because it's the real-time tracking foundation.

---

## Technical Mechanics & API Overview

### Step 1 API: Finding points to track

Before you can track motion, you need to pick *which* points to follow. Flat surfaces and smooth gradients are impossible to track (there's nothing distinctive). You need corners and high-contrast edges.

```python
feature_params = dict(
    maxCorners=100,       # int: max number of corners to return
    qualityLevel=0.3,     # float (0.0-1.0): minimum accepted corner quality relative to the strongest corner found
    minDistance=7,         # int: minimum pixel distance between returned corners (prevents clustering)
    blockSize=7           # int: size of the neighborhood used to compute the corner score for each pixel
)

# Returns: ndarray of shape (N, 1, 2) — N detected points, each as [x, y]
# The extra middle dimension (1) is an OpenCV convention for point arrays.
# mask=None means "search the entire image." You could pass a binary mask to restrict the search area.
p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
```

`qualityLevel` works like this: OpenCV scores every pixel's "corner-ness." The strongest corner gets score `S_max`. Any corner with score below `qualityLevel * S_max` is rejected. So `0.3` means "only keep corners that are at least 30% as strong as the best one."

### Step 2 API: Tracking those points in the next frame

Given the previous frame, the current frame, and the points you found, Lucas-Kanade tries to find where each point moved to.

```python
lk_params = dict(
    winSize=(15, 15),     # tuple: size of the search window around each point. Larger = handles faster motion but more expensive.
    maxLevel=2,           # int: number of image pyramid levels. 0 = original resolution only.
                          #   Pyramids downsample the image to handle large displacements that
                          #   would violate the "small motion" assumption at full resolution.
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                          # tuple: stopping criteria for the iterative search algorithm.
                          #   TERM_CRITERIA_COUNT: stop after 10 iterations max.
                          #   TERM_CRITERIA_EPS: stop if the search window moves less than 0.03 pixels.
                          #   The | combines both — whichever triggers first.
)

# p0: the points from the previous frame (shape: N, 1, 2)
# None: we don't provide an initial guess for the new positions (let OpenCV compute from scratch)
p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
```

**Return values:**

| Variable | Type | Shape | Meaning |
|----------|------|-------|---------|
| `p1` | ndarray | `(N, 1, 2)` | New `[x, y]` positions of each tracked point in `curr_gray` |
| `st` | ndarray | `(N, 1)` | Status: `1` = point was successfully tracked, `0` = point was lost (went off-screen, occluded, etc.) |
| `err` | ndarray | `(N, 1)` | Per-point error measure. Higher = less confident in the match. Often ignored in basic implementations. |

### Drawing API: Lines and circles for motion trails

```python
# cv2.line(image, start_point, end_point, color, thickness)
#   image: the numpy array to draw on (modified in-place)
#   start_point / end_point: tuple of (x, y) ints
#   color: BGR tuple, e.g. (0, 255, 0) for green
#   thickness: int, line width in pixels
cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.circle(image, center, radius, color, thickness)
#   center: tuple of (x, y) ints
#   radius: int, in pixels
#   thickness: -1 fills the circle solid
cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
```

### Combining images: `cv2.add`

```python
# cv2.add(src1, src2) -> dst
#   Performs per-pixel addition with saturation (values cap at 255, never overflow/wrap).
#   Both arrays must have the same shape and type.
#   This is how you overlay the trail drawing (mask) on top of the live video (frame).
output = cv2.add(frame, mask)
```

---

## Step-by-Step Task: Build `week2_optical_flow.py`

Create `month1/week2/week2_optical_flow.py`.

### 1. Setup: capture first frame and detect initial points

Open your camera stream. Read the very first frame before entering the loop — this becomes your "previous frame" reference. Convert it to grayscale and run `goodFeaturesToTrack` on it to get the initial set of points (`p0`).

Also create a blank canvas for drawing trails:
```python
mask = np.zeros_like(first_frame)
```

This creates a black image with the exact same dimensions and dtype as your first frame. You'll draw colored lines on this canvas. It stays persistent across loop iterations so the trails accumulate and don't disappear each frame.

### 2. The main loop: track, filter, draw, update

Inside your `while True:` loop:

**a) Read the current frame and convert to grayscale.**

**b) Run `calcOpticalFlowPyrLK`** with `prev_gray`, `curr_gray`, and `p0`.

**c) Filter to only valid points.** `st` tells you which points were successfully tracked. Use it as a boolean index:

```python
good_new = p1[st == 1]   # shape: (M, 2) — only the successfully tracked points from the current frame
good_old = p0[st == 1]   # shape: (M, 2) — their corresponding positions from the previous frame
```

Note: `st == 1` produces a boolean array. Using it as an index is NumPy boolean indexing — equivalent to `.filter()` in JS. The `[st == 1]` operation also flattens away that extra middle dimension, so the shape goes from `(N, 1, 2)` to `(M, 2)`.

**d) Loop over the point pairs and draw.** `zip(good_new, good_old)` gives you `(current_point, previous_point)` pairs. For each pair:

- Extract `(x, y)` coordinates. These come as floats from OpenCV, but drawing functions need ints. Convert with `int()`.
- Draw a line on `mask` from the old position to the new position. This creates the motion trail.
- Draw a small filled circle on the current `frame` at the new position. This shows where the point is *right now*.

Why draw trails on `mask` and dots on `frame`? Because `frame` gets overwritten every iteration (it's the live camera feed), but `mask` persists — so trails accumulate over time. Then you combine them.

**e) Combine and display:**
```python
output = cv2.add(frame, mask)
cv2.imshow("Optical Flow", output)
```

**f) Update for next iteration.** The current frame becomes the previous frame, and the current points become the previous points:

```python
prev_gray = curr_gray.copy()
p0 = good_new.reshape(-1, 1, 2)
```

Why `reshape(-1, 1, 2)`? `calcOpticalFlowPyrLK` expects input points in shape `(N, 1, 2)`. After the boolean indexing in step (c), `good_new` has shape `(M, 2)`. The reshape adds back that required middle dimension. `-1` means "calculate N automatically from the data."

**g) Re-detect corners periodically.** Points get lost over time (they go off-screen, get occluded, or the tracked patch changes too much). If you don't refresh, you'll eventually have zero points left. Add a check: if `len(p0) < 10`, re-run `goodFeaturesToTrack` on the current grayscale frame to get fresh points.

---

## Checkpoint Questions

1. Why can't optical flow track points on a smooth, plain white wall?
2. What does `st == 1` indicate in the output of `calcOpticalFlowPyrLK`?
3. Why do we draw trails on a separate `mask` image instead of directly on `frame`?
4. What would happen if you never re-detected corners inside the loop?

---

## Challenge (No Guidance)

**Direction Indicator**

Modify your optical flow script (or save as `week2_challenge_direction.py`) to:

1. Calculate the average displacement `mean_dx` and `mean_dy` across all active points in each frame.
2. Compute overall magnitude: `magnitude = np.sqrt(mean_dx**2 + mean_dy**2)`.
3. If `magnitude > 3.0` (to ignore static camera jitter), display a text overlay indicating the primary direction: `"Moving RIGHT"`, `"Moving LEFT"`, `"Moving UP"`, or `"Moving DOWN"`.
4. Optionally draw a large arrow on screen showing the overall movement vector using `cv2.arrowedLine`.

---

## Supplemental Reading

**For interviews:**
- **The 3 Core Assumptions of Lucas-Kanade** (these come up in interviews verbatim):
  1. *Brightness Constancy:* Pixel intensity of a tracked point doesn't change drastically between frames.
  2. *Small Motion:* Points don't teleport long distances between consecutive frames. Image pyramids handle larger displacements by downsampling first.
  3. *Spatial Coherence:* Neighboring pixels move together in similar directions.
- The full pipeline (goodFeaturesToTrack + calcOpticalFlowPyrLK) is often called a **KLT (Kanade-Lucas-Tomasi) tracker** in literature and interviews.

**For production context:**
- **Bridging detection gaps:** Deep learning detectors like YOLO run at 30-60 FPS. KLT optical flow can run at 200+ FPS on embedded hardware (NVIDIA Jetson, drone flight controllers) to interpolate target positions between YOLO inference frames.
- **Visual Odometry & SLAM:** Optical flow is a core component of Visual Odometry — allowing autonomous drones to estimate their own speed and heading when GPS is jammed or unavailable.
