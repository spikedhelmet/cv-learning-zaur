# Week 2, Task 3: Edge Detection (Canny)

## The Concept

Edge detection finds the boundaries of objects in an image — the sharp transitions where pixel intensity changes rapidly. If you have a dark drone against a bright sky, the edge detector will highlight the drone's outline.

Unlike background subtraction (which needs motion), edge detection works on **single frames**. It doesn't care if the object is moving or stationary. This makes it useful for a different purpose: understanding the *shape* of objects, detecting structural features like lines and corners, and as a preprocessing step for more advanced algorithms.

### How Canny Works (4 Steps Under the Hood)

The Canny edge detector is the most widely used edge detector in CV. It runs four steps internally:

1. **Gaussian Blur** — Smooths the image to reduce noise (noise creates fake edges).
2. **Gradient Calculation** — Computes how fast pixel intensity changes at each point, and in which direction. Strong gradients = edges.
3. **Non-Maximum Suppression** — Thins the edges down to 1-pixel-wide lines by keeping only the local maximum gradient points.
4. **Hysteresis Thresholding** — Uses two thresholds (low and high) to decide what counts as an edge:
   - Gradient **above high threshold** → definitely an edge
   - Gradient **below low threshold** → definitely not an edge
   - Gradient **between the two** → only kept if it's connected to a "definitely an edge" pixel

### The Function

```python
edges = cv2.Canny(image, threshold1, threshold2)
```
- **`cv2.Canny(image, threshold1, threshold2)`** — Applies the Canny edge detector to a grayscale image.
  - `image` — Input image. Should be grayscale (single channel). If you pass a color image, it will still work but results are less predictable.
  - `threshold1` (float) — The **low** threshold for hysteresis. Edges with gradient below this are discarded.
  - `threshold2` (float) — The **high** threshold for hysteresis. Edges with gradient above this are kept unconditionally.
  - Returns a binary `numpy.ndarray` of same dimensions as input, dtype `uint8`. Pixels are `255` (edge) or `0` (not edge).

**Rule of thumb:** `threshold2` should be 2–3x `threshold1`. Common starting values: `threshold1=50, threshold2=150`.

### Gaussian Blur (Pre-processing)

Canny has a built-in blur step, but it's often better to apply your own blur first for more control:

```python
blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
```
- **`cv2.GaussianBlur(src, ksize, sigmaX)`** — Applies a Gaussian (bell-curve shaped) blur to smooth the image.
  - `src` — Input image.
  - `ksize` — Kernel size as `(width, height)`. Must be **odd** numbers (3, 5, 7, etc.). Larger = more blur.
  - `sigmaX` (float) — Standard deviation of the Gaussian in the X direction. If `0`, OpenCV auto-calculates it from `ksize`. In practice, just pass `0` and control blur strength via `ksize`.
  - Returns a blurred image with same shape and dtype as input.
  - **TypeScript analogy:** This is the `filter: blur(Xpx)` CSS property, but operating directly on the pixel matrix. A `(5, 5)` kernel is like `blur(2.5px)`.

## Guided Task

Create `week2_canny.py`:

1. Capture frames from your webcam, resize to `640x480`.
2. Convert to grayscale.
3. Apply Gaussian blur with a `(5, 5)` kernel.
4. Apply Canny edge detection with `threshold1=50, threshold2=150`.
5. Display the original frame and the edge-detected frame side by side using `np.hstack` (remember to convert the edges to BGR first so the dimensions match).

6. **Make it interactive with trackbars.** This is where it gets interesting. OpenCV has a built-in UI element called a **trackbar** (slider) that lets you adjust values in real-time:

   ```python
   cv2.namedWindow("Canny Tuner")
   cv2.createTrackbar("Low", "Canny Tuner", 50, 255, lambda x: None)
   cv2.createTrackbar("High", "Canny Tuner", 150, 255, lambda x: None)
   ```
   - **`cv2.namedWindow(winname)`** — Creates a named window. You need to create the window first before attaching trackbars to it.
     - `winname` (str) — The window name. Must match the name you use in `cv2.imshow()` later.
   - **`cv2.createTrackbar(trackbar_name, window_name, value, max_value, callback)`** — Creates a slider control attached to a window.
     - `trackbar_name` (str) — Label shown next to the slider.
     - `window_name` (str) — Which window to attach it to (must already exist via `namedWindow`).
     - `value` (int) — The initial/default position of the slider.
     - `max_value` (int) — The maximum value the slider can reach.
     - `callback` — A function called every time the slider moves. We pass `lambda x: None` (a do-nothing function) because we'll read the value manually instead.
     - **TypeScript analogy:** This is like creating an `<input type="range" min="0" max="255" value="50">` element and reading its value with `element.value`.

   Then inside the loop, read the current trackbar positions:
   ```python
   low = cv2.getTrackbarPos("Low", "Canny Tuner")
   high = cv2.getTrackbarPos("High", "Canny Tuner")
   edges = cv2.Canny(blurred, low, high)
   ```
   - **`cv2.getTrackbarPos(trackbar_name, window_name)`** — Returns the current integer value of the specified trackbar.

7. Display the result in the `"Canny Tuner"` window (same name you created).

**What to observe:**
- Drag the **Low** slider up → fewer weak edges survive, cleaner output but you might lose fine details.
- Drag the **High** slider down → more edges are accepted as "definite," output gets busier.
- Find the sweet spot where you can clearly see the outline of objects in your room without too much noise.

---

## 🧩 Challenge (No Guidance)

**Combine Canny edges with your motion detector.**

Modify your `week2_motion_detector.py` (or create a new file) to:
1. Run background subtraction to find motion regions.
2. Run Canny edge detection on the same frame.
3. Use `cv2.bitwise_and()` to combine the motion mask with the edge mask — this should give you edges *only* where motion is happening.
4. Find contours on the combined result and draw bounding boxes.

This is a basic form of **multi-cue detection** — using two independent signals (motion + edges) to improve detection accuracy. Save it as `week2_challenge_motion_edges.py`.

---

## 📚 Supplemental Reading

**For interviews:** Canny is the go-to answer for "how would you detect edges in an image?" Know the four internal steps (blur, gradient, non-max suppression, hysteresis). Interviewers may also ask about Sobel filters (which compute the gradients that Canny uses internally) — you don't need to implement them, but knowing that Canny uses Sobel under the hood shows depth.

**For production context:** In modern deep-learning pipelines, explicit edge detection is rarely used as a primary detection method (YOLO handles it implicitly). But it's still used for specific tasks: lane detection in autonomous vehicles, document scanning (finding the rectangle of a page), and as a feature extraction step in some classical pipelines.
