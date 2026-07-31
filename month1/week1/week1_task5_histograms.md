# Week 1, Task 5: Histograms & Contrast Enhancement

## The Concept

A histogram is just a bar chart of pixel intensities. For a grayscale image, it counts how many pixels have intensity 0 (pure black), how many have intensity 1, all the way to 255 (pure white).

**Why this matters for defense CV:**
Your drone camera might be pointing at a bright sky, making the drone appear as a faint dark blob with very little contrast. Or it's night, and the entire image is clustered in the low-intensity range. Histogram equalization redistributes the pixel intensities across the full 0-255 range, boosting contrast so that subtle features become visible to both human operators and detection models.

**TypeScript analogy:** Think of it like a data visualization. You have an array of 256 buckets, and you're counting how many pixels fall into each bucket. It's literally `Array(256).fill(0)` and then incrementing `counts[pixelValue]++` for every pixel.

## Guided Task

Create `week1_histograms.py`. Using your phone camera feed:

1. Capture frames, resize to 640x480 (not square this time — keep the natural aspect ratio).
2. Convert each frame to grayscale.
3. Calculate the histogram using `cv2.calcHist()`:
   ```python
   hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
   ```
   - `[gray_frame]` — the image (must be in a list)
   - `[0]` — channel index (0 for grayscale, or 0/1/2 for B/G/R)
   - `None` — no mask (analyze the entire image)
   - `[256]` — number of bins
   - `[0, 256]` — the range of values

4. **Visualize the histogram on a canvas.** Create a blank 256x200 white canvas using `np.ones((200, 256), dtype=np.uint8) * 255`. Then normalize the histogram values to fit within 200 pixels tall:
   ```python
   cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
   ```
   Loop through all 256 bins and draw a vertical line for each:
   ```python
   for i in range(256):
       cv2.line(hist_canvas, (i, 200), (i, 200 - int(hist[i])), 0, 1)
   ```

5. Apply histogram equalization to the grayscale frame:
   ```python
   equalized = cv2.equalizeHist(gray_frame)
   ```

6. Display three windows side by side using `np.hstack`:
   - The original grayscale frame (resized to 256 wide to match the histogram)
   - The histogram canvas
   - The equalized frame (also resized to 256 wide)

7. **Experiment:** Cover your camera lens partially to create a dark image, then uncover it. Watch how the histogram shifts and how equalization reacts.

---

## 🧩 Challenge (No Guidance)

**CLAHE — Adaptive Histogram Equalization:**

Regular `equalizeHist` operates globally on the entire image, which can blow out already-bright areas while trying to fix dark regions. CLAHE (Contrast Limited Adaptive Histogram Equalization) divides the image into small tiles and equalizes each one independently, then blends them together.

Look up `cv2.createCLAHE()` in the OpenCV docs and apply it to your grayscale feed. Display the regular equalized frame next to the CLAHE result and observe the difference, especially when your scene has both very bright and very dark regions.

Save it as part of the same `week1_histograms.py` file.

---

## 📚 Supplemental Reading

**For interviews:** "How would you handle poor contrast in your detection input?" CLAHE is the standard answer for preprocessing before feeding frames into a model. It's also used heavily in medical imaging (X-rays, CT scans).

**For deeper understanding:** The Wikipedia article on [Histogram Equalization](https://en.wikipedia.org/wiki/Histogram_equalization) has clean visual diagrams showing how the cumulative distribution function (CDF) is used to remap intensities. You don't need to memorize the math, but understanding *why* equalization makes dark images brighter (it stretches compressed intensity ranges) is useful for interviews.
