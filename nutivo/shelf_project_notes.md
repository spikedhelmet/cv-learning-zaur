# Supermarket Shelf Scanning Project

## The Problem
Scanning a supermarket shelf and providing a health score for each individual item, integrated with an existing app that already provides health scores based on barcode scans.

## Why a Standard YOLO Approach Fails
Training a standard object detection model (like YOLO) to classify each specific product as a separate class is not viable:
- A typical supermarket has 10,000 - 50,000 unique SKUs.
- It would require thousands of annotated images *per product*.
- The model would need retraining every time a new product is added.
- A model with 50,000 classes performs poorly.

Product identification on shelves is a **retrieval** problem, not a classification problem.

## Three Potential Approaches

### 1. The Barcode-First Approach (Recommended)
This approach reuses the existing infrastructure (barcode -> health score database).

* **Stage 1: Detection** - Train YOLO with a single class (`"product"`) to detect rectangular product-shaped regions on shelves. This determines *where* products are.
* **Stage 2: Barcode Scanning** - For each cropped product region, attempt to detect and read the barcode directly using libraries like `pyzbar` or `ZXing`.
* **Stage 3: Lookup** - Query the existing database with the scanned barcode to retrieve the health score.

**Pros:**
- Reuses existing database and logic.
- Easiest and fastest to build.

**Cons:**
- Barcodes are not always visible (facing sideways, covered by tags, on the back of the package). Hit rate will not be 100%.

### 2. The Embedding Matching Approach
This approach uses visual similarity.

* **Stage 1: Detection** - Generic product detection (YOLO, 1 class) as in the previous approach.
* **Stage 2: Embedding Generation** - Pass the cropped product region through a pre-trained feature extractor (e.g., CLIP, ResNet) to generate an embedding vector.
* **Stage 3: Matching** - Compare this vector against a reference database of product embeddings using cosine similarity. The closest match identifies the product.

**Pros:**
- Works even if the barcode isn't visible, as long as the packaging is distinct.
- No retraining required when adding new products (just add the new embedding to the database).

**Cons:**
- Requires building a reference database: one clean photo per product. This is a significant logistical challenge (though scraping online grocery stores or using Open Food Facts could help).

### 3. The OCR Approach
This approach relies on reading text on the packaging.

* **Stage 1: Detection** - Generic product detection.
* **Stage 2: OCR** - Use libraries like EasyOCR or PaddleOCR to read text on the product (brand, name).
* **Stage 3: Fuzzy Matching** - Match the extracted text against the product database.

**Pros:**
- Product names are usually visible on the front face.
- No reference photos needed.

**Cons:**
- **Distance & Resolution:** OCR struggles with small text from 1-2 meters away.
- **Text Chaos:** Which text identifies the product? Packages contain a mix of brand, flavor, weight, and promotional text.
- **Multilingual:** Azerbaijani shelves contain text in Azerbaijani, Russian, Turkish, English, etc.
- **Fuzzy Matching:** Extremely complex to robustly match imperfect OCR output to database entries.

## Getting Started: Domain Adaptation for Detection
For Stage 1 (Detection) in any of these approaches, you need a generic product detector.

1. **Base Dataset:** Start by fine-tuning YOLO on the **SKU-110K** dataset (a free, public dataset with ~11K shelf images and 1.7M annotated product bounding boxes).
2. **Local Testing:** Test the fine-tuned model in local Azerbaijani stores (e.g., Bravo, Araz).
3. **Domain Adaptation:** Annotate only the failure cases (e.g., specific local packaging shapes, unique lighting, or shelf styles) and add them to the training set. This requires far fewer images (100-200) than starting from scratch.
