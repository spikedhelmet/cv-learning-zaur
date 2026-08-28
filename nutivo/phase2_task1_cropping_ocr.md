# Phase 1.3 + 2.1 — Product Cropping & OCR Fundamentals

---

## Goal

Build the bridge between detection and identification. Your YOLO model can find *where* products are on a shelf. Now you need to crop each one out and attempt to read the text on it. By the end of this task, you'll have a folder full of individual product crops and a working OCR pipeline that reads text from them.

---

## Concept: The Crop-Then-Identify Pattern

This is a fundamental pattern in applied CV, used everywhere from license plate readers to document scanning to retail analytics:

1. **Detect** — Find regions of interest in a large image (your YOLO model)
2. **Crop** — Extract each region as its own small image
3. **Identify** — Run a specialized model on each crop (OCR, embedding, barcode reader, etc.)

The reason you crop first instead of running OCR on the entire shelf image: OCR models are designed for images where text is prominent. A full shelf image has dozens of products, price tags, shelf labels, and promotional material. If you feed the whole thing to OCR, you'll get a chaotic mess of text with no way to know which text belongs to which product. Cropping isolates each product so OCR only sees one item at a time.

---

## Technical Mechanics: How OCR Works

OCR is actually two separate neural networks working in sequence:

### Stage 1: Text Detection
The first network scans the image and finds *where* text exists. It outputs bounding boxes around each text region. This is similar to what YOLO does for objects, but specifically tuned for text (which has unique properties: horizontal lines, high contrast, specific aspect ratios).

### Stage 2: Text Recognition  
The second network takes each cropped text region and reads *what* it says. It converts the pixel pattern into a string of characters. Under the hood, this uses a sequence model (similar to speech recognition) that slides across the text image from left to right, predicting one character at a time.

### EasyOCR
EasyOCR bundles both stages together. You give it an image, it returns a list of:
- The bounding box of each text region
- The recognized text string
- A confidence score (0.0 to 1.0)

It supports 80+ languages out of the box, including Azerbaijani, Russian, Turkish, and English — which is exactly what you need for Azerbaijani supermarket shelves.

---

## Step-by-Step Task

### 1. Download Your Trained Model

If you trained on Ultralytics HUB, download your `best.pt` file and place it in your project. For example:
```
nutivo/models/shelf_detector.pt
```

### 2. Write the Crop Script

Create `nutivo/crop_products.py`. This script will:
1. Load your trained shelf detector
2. Load a shelf image
3. Run detection
4. Crop each detected product region
5. Save each crop as a separate image file

The key OpenCV operation here is **array slicing** — you already know this from Week 1. If a bounding box is at `(x1, y1)` to `(x2, y2)`, then:
```python
crop = image[y1:y2, x1:x2]
```

Remember: NumPy arrays are `[row, column]` which means `[y, x]`, not `[x, y]`.

The detection results from `model.predict()` give you boxes in `xyxy` format (x1, y1, x2, y2) as floats. You need to:
- Convert them to integers (pixel coordinates)
- Clamp them to image boundaries (make sure x1 >= 0, y2 <= image height, etc.)
- Skip any crops that are too small (e.g., less than 20x20 pixels — probably noise)

Run this on your 4 shelf test photos. You should end up with a folder containing hundreds of individual product images.

### 3. Install EasyOCR

```bash
pip install easyocr
```

Note: EasyOCR will download language models on first use (~100MB per language). This is a one-time download.

### 4. Test OCR on a Single Product Crop

Create `nutivo/test_ocr.py`. Start simple:

```python
import easyocr

reader = easyocr.Reader(['en', 'ru', 'az'])  # English, Russian, Azerbaijani

results = reader.readtext('path/to/one/crop.jpg')

for (bbox, text, confidence) in results:
    print(f"Text: {text:30s} | Confidence: {confidence:.2f}")
```

Run this on a few different crops:
- A crop where the product name is clearly visible and facing the camera
- A crop where the product is at an angle
- A crop where the product is small or far from the camera
- A crop with Cyrillic/Russian text

Observe how the text quality and confidence change across these cases. This gives you an intuition for OCR's limitations before you build the full pipeline.

### 5. Visualize OCR Results on the Crop

EasyOCR returns bounding boxes for each text region it finds. Draw these on the crop image so you can see exactly what the model is reading:

```python
import cv2

img = cv2.imread('path/to/crop.jpg')
for (bbox, text, confidence) in results:
    # bbox is a list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    top_left = tuple(int(v) for v in bbox[0])
    bottom_right = tuple(int(v) for v in bbox[2])
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("OCR", img)
cv2.waitKey(0)
```

### 6. Run OCR on All Crops (Batch)

Loop through your entire crops folder and run OCR on every image. For each crop, concatenate all the recognized text into a single string. Print it alongside the filename so you can manually evaluate how well OCR is doing:

```
crop_001.jpg -> "Coca Cola Classic 330ml"
crop_002.jpg -> "Lay's Paprika 140g"
crop_003.jpg -> "dfs kj3" (garbage — OCR failed on this one)
```

Count how many crops produce readable, useful text vs. garbage. This gives you your first "OCR hit rate" estimate.

---

## Checkpoint Questions

1. Why do we crop each product before running OCR, instead of running OCR on the entire shelf image?
2. EasyOCR returns a confidence score for each text region. If you get a result with confidence 0.15, should you trust it? What threshold would you set?
3. Your YOLO detector outputs box coordinates as floats (e.g., `x1=234.7`). Why do you need to convert these to integers before using them to crop a NumPy array?
4. What happens if a detected box extends beyond the image boundary (e.g., `x2 = 1300` on a 1280-pixel-wide image)? How would you handle this?

---

## Challenge (No Guidance)

**OCR Quality Report**

Run OCR on all your crops and write a script that generates a report:
- Total crops processed
- Crops where OCR returned at least one text result with confidence > 0.5
- Crops where OCR returned nothing useful
- The 10 highest-confidence text results
- The 10 lowest-confidence text results (that still had *some* text)
- Average number of text regions detected per crop

This report will tell you whether OCR is viable for your use case or whether you should pivot to embeddings.

---

## Supplemental Reading

**For interviews:**
- **"What is OCR and how does it work?"** — "OCR is a two-stage pipeline: text detection (finding where text regions are in an image using a CNN) followed by text recognition (converting each detected text region into a character string using a sequence model, typically a CRNN with CTC loss)."
- **"When does OCR fail?"** — "Low resolution, extreme angles, motion blur, unusual fonts, and overlapping text. OCR also struggles with very small text at a distance, which is why high-resolution input matters."

**For production context:**
- **PaddleOCR** is the main alternative to EasyOCR. It's faster and often more accurate, especially for Asian languages. If EasyOCR's speed or accuracy isn't good enough, PaddleOCR is the next step.
- **Document AI vs. Scene Text:** OCR models trained on documents (flat, high-contrast, standard fonts) perform poorly on "scene text" (text in natural images — products, signs, etc.). EasyOCR and PaddleOCR are designed for scene text.

**External resources:**
- EasyOCR GitHub: https://github.com/JaidedAI/EasyOCR — Supported languages, examples, and API reference.
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR — Higher-performance alternative if you need it later.
