# Week 2 — Task 4: Sparse Optical Flow (Lucas-Kanade)

---

## 🎯 Goal
Understand what **Optical Flow** is, how it differs from background subtraction, and build a real-time feature tracking script using **Lucas-Kanade Optical Flow** (`cv2.calcOpticalFlowPyrLK`).

---

## 💡 Concept: Motion vs. Velocity Vector

So far, you've used **Background Subtraction** to find *where* pixels changed between frames. 

However, Background Subtraction only tells you:  
> *"Something moved in this region."*

**Optical Flow** takes it a step further and tells you:  
> *"This specific point moved 12 pixels to the right and 5 pixels up."*

It calculates **velocity vectors** $(dx, dy)$ for points between consecutive frames $t$ and $t+1$.

### 🛡️ Defense Context
In counter-drone C2 (Command and Control) systems:
* **Background Subtraction** detects a target entering the sky.
* **Optical Flow** estimates the target's **trajectory and speed vector** across the screen, allowing the system to predict where the target will be 1 second from now.

### 🌐 Web / TS Analogy
Think of Optical Flow like tracking drag velocity in web animations:
* When a user drags an element on screen, you calculate `deltaX = currentX - prevX` and `deltaY = currentY - prevY` on every `onPointerMove` event to determine drag direction and velocity.
* Optical Flow does the exact same calculation automatically for pixels between video frames!

---

## 🔍 Sparse vs. Dense Optical Flow

1. **Sparse Optical Flow (Lucas-Kanade):**  
   Tracks a select set of "interesting" points (corners, high-contrast features).  
   * **Pros:** Extremely fast, lightweight, runs smoothly on edge devices / low-power hardware.
   * **Cons:** Only tracks specific points, not the entire object.

2. **Dense Optical Flow (Farneback):**  
   Calculates motion vectors for **every single pixel** in the image.  
   * **Pros:** Complete motion field visualization.
   * **Cons:** Computationally expensive.

We will focus on **Sparse Lucas-Kanade** as it's the standard real-time tracking foundation.

---

## ⚙️ The 2-Step Lucas-Kanade Pipeline

### Step 1: Find Good Features to Track (`cv2.goodFeaturesToTrack`)
Optical flow needs strong, distinct points to track (like sharp corners or high-contrast edges). Flat surfaces or smooth gradients cannot be tracked reliably.

```python
feature_params = dict(
    maxCorners=100,       # Maximum number of corners to track
    qualityLevel=0.3,     # Minimal accepted quality of image corners (0.0 - 1.0)
    minDistance=7,        # Minimum distance between returned corners
    blockSize=7
)

# Detect corners on grayscale frame
p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
```

---

### Step 2: Track Points in Next Frame (`cv2.calcOpticalFlowPyrLK`)
Pass the previous frame, current frame, and the detected corners `p0` into `cv2.calcOpticalFlowPyrLK`:

```python
lk_params = dict(
    winSize=(15, 15),     # Size of the search window at each pyramid level
    maxLevel=2,           # Pyramid levels (allows tracking larger movements)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p0, None, **lk_params)
```

**Understanding the outputs:**
* `p1`: The new positions of the tracked points in `gray_curr`.
* `st`: Status vector (`1` if flow was found for the point, `0` if lost/failed).
* `err`: Error vector measuring tracking confidence.

---

## 🛠️ Step-by-Step Task: Build `week2_optical_flow.py`

Create a new file `month1/week2/week2_optical_flow.py` and implement the following:

1. Open webcam stream `cv2.VideoCapture(0)` and read the first frame.
2. Convert first frame to grayscale (`prev_gray`) and detect initial feature points (`p0`).
3. Create a blank mask canvas `mask = np.zeros_like(first_frame)` to draw motion trails on.
4. Enter the `while True:` loop:
   * Read current frame (`frame`) and convert to grayscale (`gray`).
   * Calculate optical flow between `prev_gray` and `gray` using `cv2.calcOpticalFlowPyrLK`.
   * Filter only valid points where `st == 1`.
   * Loop over valid points, drawing a line on `mask` from `(prev_x, prev_y)` to `(curr_x, curr_y)` and a small circle at `(curr_x, curr_y)`.
   * Combine `frame` and `mask` using `cv2.add(frame, mask)` to display trails over the video.
   * Update `prev_gray = gray.copy()` and `p0 = good_new.reshape(-1, 1, 2)`.
   * Every N frames (or if `len(p0) < 10`), re-detect corners with `cv2.goodFeaturesToTrack` so you don't lose all track points over time!

---

## ❓ Checkpoint Questions
Before writing code, answer these in your head:
1. Why can't optical flow track points on a smooth, plain white wall?
2. What does `st == 1` indicate in `cv2.calcOpticalFlowPyrLK`?

Give `week2_optical_flow.py` a try!
