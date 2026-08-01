# 8-Week Applied Computer Vision Roadmap

**Goal:** Build a portfolio-ready Defense C2 Dashboard with a CV backend. Prepare for interviews at a defense company focused on counter-drone systems.

**Start Date:** 2026-07-29 (Day 1)

---

## Week 1: OpenCV & Image Processing Fundamentals *(Complete)*
**Status:** Complete

- [x] Environment setup (Python, NumPy, OpenCV, matplotlib)
- [x] Images as NumPy matrices, BGR color space, drawing primitives
- [x] Live webcam capture (`VideoCapture`), frame loop, `waitKey` idioms
- [x] Preprocessing pipeline: grayscale conversion, resize, crop
- [x] Writing video to disk with `VideoWriter`
- [x] Color spaces: BGR vs HSV vs Grayscale — when and why each matters
- [x] Thresholding: binary, adaptive, Otsu's method
- [x] Morphological operations: erosion, dilation, opening, closing
- [x] Histograms and histogram equalization (contrast enhancement, CLAHE)

**Deliverable:** Script that takes a noisy/low-contrast image and produces a clean binary mask of a target region.

---

## Week 2: Motion Detection & Classical Object Detection *(In Progress)*
- [x] Background subtraction (MOG2, KNN) — detecting moving objects in a static scene
- [x] Contour detection: `findContours`, contour area filtering, bounding rectangles
- [x] Build a basic motion detector: highlight moving objects with bounding boxes
- [ ] Edge detection (Canny) and when it's useful
- [ ] Optical flow basics (sparse with Lucas-Kanade) — tracking point motion between frames
- [ ] ROI (Region of Interest) selection and masking

**Deliverable:** A live motion detection script that draws bounding boxes around moving objects in your phone's camera feed.

---

## Week 3: Introduction to Deep Learning for CV
- [ ] What neural networks actually do (conceptual, no math rabbit holes)
- [ ] Classification vs Detection vs Segmentation — the three tasks
- [ ] What a "model" is: weights file, architecture, inference
- [ ] Install `ultralytics`, run pre-trained YOLOv8/11 on sample images
- [ ] Understand YOLO output: bounding boxes, class IDs, confidence scores
- [ ] Run YOLO on your live phone camera feed
- [ ] FPS measurement and basic performance awareness

**Deliverable:** Live YOLO detection on your phone camera feed with FPS counter overlay.

---

## Week 4: Datasets, Annotation & Roboflow
- [ ] What makes a good detection dataset (diversity, balance, edge cases)
- [ ] Roboflow account setup, explore existing drone/UAV datasets
- [ ] Annotation formats: YOLO txt, COCO JSON, Pascal VOC XML
- [ ] Data augmentation: why it matters, what Roboflow provides
- [ ] Download and organize a drone detection dataset
- [ ] Dataset splits: train/val/test — what they mean and why data leakage kills you
- [ ] Understand the YOLO `data.yaml` config format

**Deliverable:** A properly structured, augmented drone dataset ready for training.

---

## Week 5: Training Custom YOLO Models
- [ ] Fine-tuning vs training from scratch — transfer learning explained
- [ ] Train YOLOv8/11 on your drone dataset (local or Google Colab if GPU needed)
- [ ] Understanding training output: loss curves, mAP, precision, recall
- [ ] What overfitting looks like and how to spot it
- [ ] Hyperparameter basics: epochs, batch size, image size, learning rate
- [ ] Validate your trained model on the test set
- [ ] Export model to ONNX format (production awareness)

**Deliverable:** A custom-trained YOLO model that detects drones, with documented training metrics.

---

## Week 6: Multi-Object Tracking
- [ ] Detection vs Tracking: why detection alone isn't enough
- [ ] Tracking algorithms overview: SORT, DeepSORT, ByteTrack, BoT-SORT
- [ ] Implement ByteTrack with your custom YOLO model
- [ ] Persistent object IDs across frames
- [ ] Handling occlusion, re-identification, track loss
- [ ] Count objects entering/exiting a zone (tripwire logic)
- [ ] Record tracked video output with IDs and trails

**Deliverable:** Live tracking pipeline that detects drones and maintains persistent IDs with visual trails.

---

## Week 7: Defense C2 Dashboard — Backend
- [ ] System architecture: CV pipeline → API → Frontend
- [ ] FastAPI backend serving detection results (JSON over WebSocket or REST)
- [ ] Stream processed video frames to the frontend (MJPEG stream or WebSocket)
- [ ] Alert system: trigger events when a drone is detected (log, API call)
- [ ] SQLite or simple file-based logging of detection events
- [ ] Geo-zone logic: define restricted areas, flag intrusions

**Deliverable:** A running FastAPI backend that processes video, runs detection+tracking, and serves results via API.

---

## Week 8: Defense C2 Dashboard — Frontend & Portfolio Polish
- [ ] React/Next.js frontend displaying the live detection feed
- [ ] Real-time detection event log (WebSocket updates)
- [ ] Dashboard UI: detection count, active tracks, alert history, zone map
- [ ] README and documentation for the portfolio repo
- [ ] Record a demo video walkthrough
- [ ] Interview prep: be able to explain every component of the pipeline
- [ ] Understand deployment context: edge devices, ONNX Runtime, TensorRT

**Deliverable:** A complete, deployable Defense C2 Dashboard with live detection, tracking, and a polished frontend.

---

## Key Concepts to Know for Interviews
These will be woven into the weekly tasks, not studied in isolation:
- **mAP (Mean Average Precision):** How detection accuracy is measured
- **Loss:** What the model optimizes during training
- **Overfitting:** When the model memorizes training data instead of learning patterns
- **Data Leakage:** When test data contaminates training, giving fake-good metrics
- **ONNX/TensorRT:** How Python prototypes become C++ production systems
- **FPS vs Accuracy tradeoff:** Nano vs Large models, when to use which
