# Shelf Scanner Project — Implementation Plan

## Overview

Replace Week 7-8's drone C2 dashboard with a supermarket shelf scanning system that identifies products and retrieves health scores. This project builds on everything you've learned (YOLO detection, tracking, training) and adds three new major skills: **OCR**, **embedding/retrieval**, and **full-stack integration**.

**Strategy:** OCR-first approach. If OCR proves unreliable (likely for distant/multilingual shelves), pivot to embedding matching. Both paths teach critical CV concepts.

---

## Phase 1: Generic Product Detection (Week 7)

You already know how to train YOLO. This phase applies that skill to a new domain.

### 1.1 — SKU-110K Dataset Setup

- [x] Download the SKU-110K dataset (public shelf detection dataset, ~11K images, 1.7M product bounding boxes)
- [x] Understand the dataset format (it uses CSV annotations, not YOLO format)
- [x] Write a conversion script: CSV → YOLO txt format
- [x] Split into train/val/test and create `data.yaml`

**New concept:** Format conversion at scale. SKU-110K uses `image_name,x1,y1,x2,y2,class,image_width,image_height` CSV rows. You'll need to convert absolute pixel coordinates to YOLO normalized format — you already did this math in Week 4.

### 1.2 — Train Product Detector

- [x] Fine-tune YOLO11n on SKU-110K (single class: `product`)
- [x] Evaluate mAP on the test set
- [x] Test on photos you take in a local store (Bravo, Araz, etc.)
- [x] Identify failure cases (local packaging shapes, lighting, shelf styles)

**Deliverable:** A YOLO model that draws bounding boxes around individual products on any shelf image.

### 1.3 — Build Reference Database from Shelf Crops (Synthetic Data)

- [ ] Take 5-10 shelf photos in a local store
- [ ] Run your trained product detector on them to automatically crop every individual product
- [ ] Save all crops to a folder (your detector does the hard work)
- [ ] Manually label each crop: create a CSV or JSON mapping `crop_filename → product_name, health_score`
- [ ] This labeled crop database will be used by both the OCR pipeline (as ground truth to validate against) and the embedding pipeline (as reference images)

**Why this works:** Instead of photographing each product individually in perfect lighting, you let YOLO cut them out of real shelf images. The crops look exactly like what the system will see in production, which actually makes matching _more_ robust than studio photos would.

---

## Phase 2: OCR Pipeline (Week 7-8)

This is the approach you want to try first. The pipeline: Detect products → Crop each product → Run OCR on the crop → Match extracted text to a database.

### 2.1 — OCR Fundamentals

- [ ] Install and test EasyOCR (supports Azerbaijani, Russian, Turkish, English out of the box)
- [ ] Understand how OCR works at a high level (text detection → text recognition)
- [ ] Run OCR on a few product images you photograph up close
- [ ] Run OCR on crops from your shelf detector — observe how quality degrades with distance

**New concepts:** Text detection vs text recognition. EasyOCR first finds _where_ text is in the image (detection), then reads _what_ it says (recognition). Two separate neural networks working in sequence.

### 2.2 — Text Extraction Pipeline

- [ ] Build the full pipeline: shelf image → YOLO detection → crop each product box → run OCR on each crop
- [ ] Filter OCR results: remove very short strings, low-confidence reads, and common noise words (e.g., "g", "ml", "NET WT")
- [ ] Experiment with preprocessing the crops before OCR (resize, sharpen, contrast enhancement — you learned these in Week 1!)

### 2.3 — Product Matching via Fuzzy Text Search

- [ ] Build a simple product database (start with 20-30 products you manually enter: name, brand, health score)
- [ ] Implement fuzzy string matching using `rapidfuzz` or `fuzzywuzzy` library
- [ ] Match OCR output to the database, return the best match + confidence score
- [ ] Evaluate: What percentage of products on a shelf can you correctly identify?

**New concept:** Fuzzy matching. OCR might read "Caca Cola 330m" instead of "Coca-Cola 330ml". Fuzzy matching algorithms (Levenshtein distance, token set ratio) can still find the correct match despite errors.

**Decision point:** After testing on 3-5 real shelf photos, evaluate the OCR hit rate. If it's below ~50%, pivot to Phase 3 (embeddings). If it's above 70%, continue refining.

**Deliverable:** A script that takes a shelf photo, detects products, reads text, and prints the best-match product name + health score for each item.

---

## Phase 3: Embedding Matching — Production Pipeline

### 3.1 — Embedding Fundamentals [DONE]

- [x] Understand what an embedding is (a fixed-length vector that captures the "essence" of an image)
- [x] Install and test CLIP (OpenAI's model that understands both images and text)
- [x] Generate embeddings for product images, measure cosine similarity between similar vs different products
- [x] Build a matcher loop that compares query crops against a reference dictionary

**New concept:** Cosine similarity. Two vectors pointing in the same direction = similar content. Cosine similarity of 1.0 = identical, 0.0 = completely unrelated.

### 3.2 — Qdrant Vector Database

- [ ] Install `qdrant-client` and `python-dotenv`
- [ ] Store Qdrant Cloud credentials in a `.env` file (never hardcode API keys)
- [ ] Create a Qdrant collection with 512-dim vectors (matching CLIP's output)
- [ ] Upload reference product embeddings with metadata (product name, health score, category)
- [ ] Query the collection with a crop embedding and retrieve the top match
- [ ] Understand the difference between exact search and approximate nearest neighbor (ANN)

**New concept:** Vector databases use indexing structures (like HNSW graphs) to search through millions of vectors in milliseconds, instead of comparing against every single one linearly.

### 3.3 — Expand the Reference Database

- [ ] Photograph or crop 30-50 unique products for the reference DB (more coverage = fewer wrong matches)
- [ ] Upload all reference embeddings to Qdrant with proper metadata
- [ ] Add multiple angles/variants per product where possible (front, side, tilted)
- [ ] Test matching accuracy with a known test set

### 3.4 — Full Matching Pipeline

- [ ] Full pipeline: shelf image → YOLO detection → crop → CLIP embedding → Qdrant search → return closest match
- [ ] Set a similarity threshold (e.g., 0.85) below which the system says "unknown product"
- [ ] Evaluate accuracy on real shelf photos and log results
- [ ] Optional: Try `clip-ViT-L-14` (768-dim, more accurate, slower) and compare against `clip-ViT-B-32`

**Deliverable:** A script that takes a shelf photo and identifies products by visual similarity via Qdrant, returning product names and health scores.

---

## Phase 4: Full-Stack Integration

### 4.1 — FastAPI Backend

- [ ] Create a FastAPI server with a `/scan` endpoint that accepts an image upload
- [ ] Run the full pipeline (YOLO detect → CLIP embed → Qdrant search) server-side
- [ ] Return JSON: list of products with bounding box positions, names, health scores, and confidence
- [ ] Add a `/products` endpoint to list all products in the Qdrant database
- [ ] Add a `/products/add` endpoint to upload a new reference product image + metadata

### 4.2 — Frontend / App Integration

- [ ] Build a simple web UI where users can upload a shelf photo (or capture from camera)
- [ ] Display the shelf image with colored overlays (green = healthy, yellow = moderate, red = unhealthy)
- [ ] Show a sidebar with the product list, names, and scores
- [ ] Document the API contract so your cousin's app can call the `/scan` endpoint directly

### 4.3 — Polish & Deploy

- [ ] Deploy the FastAPI backend (Railway, Render, or a VPS)
- [ ] Record a demo video
- [ ] Write a README documenting the architecture and results
- [ ] Prepare to explain the pipeline end-to-end in an interview

**Deliverable:** A working API + web app where you upload a shelf photo and get product identification with health scores.

---

## Updated Skills Map

| Phase         | New CV/ML Skills                                        | New Engineering Skills                        |
| ------------- | ------------------------------------------------------- | --------------------------------------------- |
| 1. Detection  | Domain adaptation, dataset conversion at scale          | Data pipeline scripting                       |
| 2. OCR        | Text detection, text recognition, preprocessing for OCR | Fuzzy matching, text processing               |
| 3. Embeddings | Feature extraction, cosine similarity, CLIP, ANN search | Qdrant, vector databases, retrieval systems   |
| 4. Full-stack | End-to-end ML pipeline                                  | FastAPI, API design, deployment, app integration |

