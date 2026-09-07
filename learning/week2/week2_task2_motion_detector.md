# Week 2, Task 2: Motion Detector with Bounding Boxes

## The Connection

In the last task, you got a raw foreground mask from background subtraction. White pixels = motion. But a mask alone isn't useful in a detection pipeline — you need to extract **where** the motion is happening and draw bounding boxes around it. You already know `findContours` and `boundingRect` from the Week 1 isolator challenge. Now you're combining them with background subtraction to build a real motion detector.

This is the exact architecture used in perimeter security systems: a stationary camera watches a restricted area, and any motion triggers an alert with a bounding box around the intruder.

## Guided Task

Create `week2_motion_detector.py`:

1. Capture frames from your webcam, resize to `640x480`.

2. Create a MOG2 background subtractor before the loop. Set `detectShadows=False` this time — shadows create gray pixels (127) in the mask which can confuse contour detection. We want a clean binary mask.

3. Inside the loop, apply the subtractor to get the foreground mask:
   ```python
   fg_mask = bg_sub.apply(frame)
   ```

4. **Clean the mask with morphology.** The raw mask will be noisy (flickering pixels, partial blobs). Apply morphological opening to kill noise, then dilation to fill gaps in the motion blobs:
   ```python
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
   cleaned = cv2.dilate(cleaned, kernel, iterations=2)
   ```
   - **`cv2.getStructuringElement(shape, ksize)`** — Creates a kernel (structuring element) with a specific shape instead of a plain square.
     - `shape` — The shape of the kernel. `cv2.MORPH_ELLIPSE` creates an elliptical (round) kernel. Other options: `cv2.MORPH_RECT` (square, same as `np.ones`), `cv2.MORPH_CROSS` (plus-shaped).
     - `ksize` — Size of the kernel as `(width, height)`.
     - Returns a `numpy.ndarray` of dtype `uint8` filled with 0s and 1s in the specified shape.
   - **Why elliptical?** Round kernels produce smoother edges on the blobs compared to square kernels. For motion detection, this gives cleaner, more natural-looking contours around moving objects.

5. **Find contours** on the cleaned mask:
   ```python
   contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```
   - Note: We're using `cv2.RETR_EXTERNAL` here instead of `cv2.RETR_TREE` that you used in Week 1.
     - `RETR_TREE` retrieves *all* contours, including ones nested inside other contours (like a hole inside a shape).
     - `RETR_EXTERNAL` retrieves only the outermost contours. For motion detection, if a person is moving, we want one box around the whole person — not separate boxes around their arm, their head, and the gap between their legs.

6. **Filter and draw bounding boxes.** Loop through the contours, filter by area (skip anything smaller than 1000 pixels to ignore noise), and draw rectangles on the original color frame:
   ```python
   for contour in contours:
       if cv2.contourArea(contour) < 1000:
           continue
       x, y, w, h = cv2.boundingRect(contour)
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   ```

7. Display **two windows** side-by-side using `np.hstack`:
   - The cleaned binary mask (so you can see what the detector "sees")
   - The original color frame with bounding boxes drawn on it

   Since the mask is single-channel (grayscale) and the frame is 3-channel (BGR), you can't directly `hstack` them. Convert the mask to 3-channel first:
   ```python
   cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
   ```
   - **`cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)`** — Converts a single-channel grayscale image to a 3-channel BGR image by duplicating the gray values into all three channels. The image still *looks* grayscale, but now its array shape is `(H, W, 3)` instead of `(H, W)`, so it's compatible with `np.hstack` alongside color frames.

**What to observe:**
- Walk in front of the camera — you should get a clean green bounding box around yourself.
- Wave just your hand — a smaller box should appear around your hand only.
- If you get tons of tiny flickering boxes, increase your area threshold or make the kernel bigger.

---

## 📚 Supplemental Reading

**For interviews:** This exact pipeline (background subtraction → morphological cleanup → contour extraction → bounding rect) is the textbook answer to "how would you build a basic surveillance motion detector?" Know it cold. Interviewers will then ask about its limitations — the main one being **the camera must be stationary**. If the camera moves (e.g., on a drone or vehicle), the entire background shifts and the algorithm thinks everything is moving. That's where optical flow and deep learning come in (later this week and in Week 3).
