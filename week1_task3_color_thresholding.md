# Week 1, Task 3: Color Spaces and Thresholding

## Why This Matters
In defense CV, cameras often deal with harsh lighting, glare, fog, and IR feeds. Raw RGB/BGR pixel values are unreliable for isolating targets under changing light. Color space conversion and thresholding are the standard tools to handle this.

## Part 1: Color Spaces (BGR → HSV)

You already know BGR (OpenCV's default) and Grayscale. Now learn **HSV (Hue, Saturation, Value)**.

HSV separates *color identity* (Hue) from *brightness* (Value). This means you can detect "red objects" regardless of whether they're in sunlight or shadow — something BGR can't do reliably.

Create a file called `week1_color_spaces.py`:

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.0.120:8080/video")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a color range to isolate. This example targets blue objects.
    # HSV ranges in OpenCV: H(0-179), S(0-255), V(0-255)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a binary mask: white where blue exists, black everywhere else
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Apply the mask to the original frame to see only the blue parts
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show all three views
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Your job:**
1. Run the script and hold up a blue object (pen, phone case, anything) in front of your phone camera.
2. Observe how the mask isolates only the blue pixels.
3. Try changing the `lower_blue` / `upper_blue` values to target a different color you have nearby (red, green, yellow). You'll need to look up the HSV hue values — red is tricky because it wraps around 0/180.

---

## Part 2: Thresholding

Thresholding converts a grayscale image into a binary image (black and white only). This is how you turn messy real-world images into clean data that algorithms can process.

Add this to the same file, or create `week1_thresholding.py`:

```python
import cv2

cap = cv2.VideoCapture("http://192.168.0.120:8080/video")

while True:
    success, frame = cap.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simple binary threshold: pixels > 127 become white (255), rest become black (0)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu's method: OpenCV automatically picks the best threshold value
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold: threshold varies across the image based on local regions
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Binary (fixed 127)", binary)
    cv2.imshow("Otsu (auto)", otsu)
    cv2.imshow("Adaptive", adaptive)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Your job:**
1. Run it and move your phone around to different lighting conditions (point at a window, then at a dark corner).
2. Watch how the fixed `127` threshold breaks in different lighting, but Otsu and Adaptive handle it better.
3. Understand why adaptive thresholding matters for real-world deployments where lighting is never consistent.

---

## What to Report Back
- Did the color mask isolate the object you targeted?
- Which thresholding method handled lighting changes best?
- Any questions about when you'd use one technique over another?
