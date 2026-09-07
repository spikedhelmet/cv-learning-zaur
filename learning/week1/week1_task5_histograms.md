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

4. **Visualize the histogram on a canvas.**

   First, create a blank white canvas:

   ```python
   hist_canvas = np.ones((200, 256), dtype=np.uint8) * 255
   ```

   - **`np.ones((200, 256), ...)`** — Creates a 2D NumPy array of shape (200 rows, 256 columns) filled entirely with the value `1`. Compare this to `np.zeros()` which fills with `0`. Think of it like `new Array(200 * 256).fill(1)` in JS, but shaped as a 2D grid.
   - **`dtype=np.uint8`** — Specifies the data type of every element in the array. `uint8` = unsigned 8-bit integer, range 0–255, just like pixel values. Using the wrong dtype (e.g., `float32`) would cause OpenCV to error or display incorrectly.
   - **`* 255`** — Multiplies every element in the array by 255. So the canvas starts as all `1`s, and becomes all `255`s — i.e., pure white pixels.

   Next, normalize the histogram so its values fit within the 200-pixel height of your canvas:

   ```python
   cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
   ```

   - **`cv2.normalize(src, dst, alpha, beta, norm_type)`**
     - `src` — the input array (your raw histogram, where values could be in the thousands)
     - `dst` — the output array to write into. Passing `hist` again means "overwrite in place" — same as `arr = normalize(arr)`.
     - `alpha` — the minimum value in the output range (0 in this case)
     - `beta` — the maximum value in the output range (200 — your canvas height)
     - `norm_type` — the normalization strategy. `cv2.NORM_MINMAX` finds the current min and max, then linearly scales all values so the smallest maps to `alpha` and the largest maps to `beta`. The TypeScript analogy: `const normalized = val => (val - min) / (max - min) * 200`.

   Now draw a vertical line per bin:

   ```python
   for i in range(256):
       cv2.line(hist_canvas, (i, 200), (i, 200 - int(hist[i])), 0, 1)
   ```

   - `range(256)` — loops `i` from 0 to 255, one iteration per histogram bin (one per intensity level)
   - `(i, 200)` — the bottom of the line (x = bin index, y = bottom of canvas)
   - `(i, 200 - int(hist[i]))` — the top of the line. Subtracting from 200 flips the bar chart upright, since y=0 is at the _top_ in image coordinates
   - `0` — the color (black in grayscale)
   - `1` — line thickness of 1 pixel

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

**For deeper understanding:** The Wikipedia article on [Histogram Equalization](https://en.wikipedia.org/wiki/Histogram_equalization) has clean visual diagrams showing how the cumulative distribution function (CDF) is used to remap intensities. You don't need to memorize the math, but understanding _why_ equalization makes dark images brighter (it stretches compressed intensity ranges) is useful for interviews.
