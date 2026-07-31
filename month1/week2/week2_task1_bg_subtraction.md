# Week 2, Task 1: Background Subtraction

## Why This Matters

Everything you built in Week 1 — color masking, thresholding, morphology — works great when you know *what color* your target is. But what if you don't? What if the drone is black, or gray, or changes color depending on the lighting?

In counter-drone systems, you often don't care about color at all. You care about **motion**. A drone moves. The sky doesn't. Background subtraction exploits exactly this: it learns what the "background" looks like over time, and then anything that *doesn't match* gets flagged as "foreground" — a potential target.

## The Concept

Background subtraction algorithms maintain a **model** of the background scene. Each new frame is compared against this model. Pixels that differ significantly from the model are classified as foreground (white in the output mask). Pixels that match are background (black).

The two main algorithms in OpenCV:

### MOG2 (Mixture of Gaussians v2)
Models each pixel's history as a mixture of Gaussian distributions. Adapts to gradual lighting changes (like clouds moving across the sky). This is the most commonly used one.

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
```
- **`cv2.createBackgroundSubtractorMOG2(...)`** — Creates and returns a background subtractor object. You create it once before the loop, then call `.apply()` on every frame.
  - `history` (int, default 500) — How many recent frames the model remembers. Higher = slower to adapt to scene changes (like a person walking into frame and stopping). Lower = forgets faster, more responsive.
  - `varThreshold` (float, default 16) — How sensitive the detector is. Lower = more sensitive (catches subtle motion, but also more noise). Higher = only flags large differences.
  - `detectShadows` (bool, default True) — If `True`, the algorithm also detects shadows and marks them as gray (127) instead of white (255) in the output mask. Useful because shadows move with objects but aren't objects themselves.

### KNN (K-Nearest Neighbors)
A newer algorithm. Generally handles scenes with more dynamic backgrounds (like waving trees or water) better than MOG2.

```python
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)
```
- **`cv2.createBackgroundSubtractorKNN(...)`** — Same interface as MOG2, different algorithm under the hood.
  - `history` — Same as MOG2.
  - `dist2Threshold` (float, default 400.0) — The distance threshold for classifying foreground vs background. Similar concept to `varThreshold` in MOG2 but uses a different metric.
  - `detectShadows` — Same as MOG2.

### Applying the subtractor

Inside your frame loop, you call `.apply()` on every frame:

```python
fg_mask = bg_subtractor.apply(frame)
```
- **`.apply(frame)`** — Takes the current frame, compares it against the learned background model, updates the model, and returns a single-channel binary mask.
  - Returns a `numpy.ndarray` of shape `(Height, Width)` with dtype `uint8`.
  - Pixel values: `255` = definite foreground, `127` = shadow (if `detectShadows=True`), `0` = background.
  - Think of it like a real-time `inRange()` that auto-learns the "in-range" thresholds from the scene history, instead of you hardcoding HSV values.

## Guided Task

Create `week2_bg_subtraction.py`:

1. Capture frames from your webcam, resize to `640x480`.
2. Create **both** a MOG2 and a KNN subtractor before the loop (use default parameters to start).
3. Inside the loop, apply both subtractors to the same frame to get two separate foreground masks.
4. Stack all three horizontally using `np.hstack`: original frame (converted to grayscale so dimensions match), MOG2 mask, KNN mask.
5. Display the stacked result.

**What to observe:**
- When you first start the script, the entire frame will be white (foreground) for a few seconds. That's the algorithm learning the background — it hasn't seen enough frames yet.
- Once it stabilizes, sit still for ~5 seconds. The background model will learn your position. Now **wave your hand** — you should see just your hand light up as white foreground.
- Now **stand up and move away from the camera**. Your previous sitting position might briefly appear as a "ghost" in the MOG2 mask. That's the old background model adjusting to your absence.

6. Experiment: Try changing `history` to `100` (fast adaptation) vs `2000` (slow adaptation) on the MOG2 subtractor and observe how it changes behavior when you move.

---

## 📚 Supplemental Reading

**For interviews:** "How would you detect a moving object without knowing what it looks like?" Background subtraction is the classical answer. Interviewers will ask about its limitations too (camera must be stationary, struggles with sudden lighting changes). Know both strengths and weaknesses.

**For production context:** In real defense systems, the camera on a tower or vehicle is often stationary, making background subtraction a legitimate first-pass filter before feeding ROIs into a heavier neural network like YOLO. It's computationally cheap (runs easily at 100+ FPS) and reduces the search space dramatically.
