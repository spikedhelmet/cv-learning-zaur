# Applied Computer Vision Roadmap

**Goal:** Build a portfolio-ready Defense C2 Dashboard with a CV backend. Progress into production-grade optimization, segmentation, and multi-camera systems. Prepare for interviews at a defense company focused on counter-drone systems.

**Start Date:** 2026-07-29 (Day 1)

---

## Week 1: OpenCV & Image Processing Fundamentals _(Complete)_

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

## Week 2: Motion Detection & Classical Object Detection _(Complete)_

- [x] Background subtraction (MOG2, KNN) — detecting moving objects in a static scene
- [x] Contour detection: `findContours`, contour area filtering, bounding rectangles
- [x] Build a basic motion detector: highlight moving objects with bounding boxes
- [x] Edge detection (Canny) and when it's useful
- [x] Optical flow basics (sparse with Lucas-Kanade) — tracking point motion between frames
- [x] ROI (Region of Interest) selection and masking

**Deliverable:** A live motion detection script that draws bounding boxes around moving objects in your phone's camera feed.

---

## Week 3: Introduction to Deep Learning for CV _(Complete)_

- [x] What neural networks actually do (conceptual, no math rabbit holes)
- [x] Classification vs Detection vs Segmentation — the three tasks
- [x] What a "model" is: weights file, architecture, inference
- [x] Install `ultralytics`, run pre-trained YOLOv8/11 on sample images
- [x] Understand YOLO output: bounding boxes, class IDs, confidence scores
- [x] Run YOLO on your live phone camera feed
- [x] FPS measurement and basic performance awareness

**Deliverable:** Live YOLO detection on your phone camera feed with FPS counter overlay.

---

## Week 4: Datasets, Annotation & Roboflow _(Complete)_

- [x] What makes a good detection dataset (diversity, balance, edge cases)
- [x] Roboflow account setup, explore existing drone/UAV datasets
- [x] Annotation formats: YOLO txt, COCO JSON, Pascal VOC XML
- [x] Data augmentation: why it matters, what Roboflow provides
- [x] Download and organize a drone detection dataset
- [x] Dataset splits: train/val/test — what they mean and why data leakage kills you
- [x] Understand the YOLO `data.yaml` config format

**Deliverable:** A properly structured, augmented drone dataset ready for training.

---

## Week 5: Training Custom YOLO Models _(Complete)_

- [x] Fine-tuning vs training from scratch — transfer learning explained
- [x] Train YOLOv8/11 on your drone dataset (local or Google Colab if GPU needed)
- [x] Understanding training output: loss curves, mAP, precision, recall
- [x] What overfitting looks like and how to spot it
- [x] Hyperparameter basics: epochs, batch size, image size, learning rate
- [x] Validate your trained model on the test set
- [x] Export model to ONNX format (production awareness)

**Deliverable:** A custom-trained YOLO model that detects drones, with documented training metrics.

---

## Week 6: Multi-Object Tracking _(Complete)_

- [x] Detection vs Tracking: why detection alone isn't enough
- [x] Tracking algorithms overview: SORT, DeepSORT, ByteTrack, BoT-SORT
- [x] Implement ByteTrack with your custom YOLO model
- [x] Persistent object IDs across frames
- [x] Handling occlusion, re-identification, track loss
- [x] Count objects entering/exiting a zone (tripwire logic)
- [x] Record tracked video output with IDs and trails

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

## Week 9: Model Optimization & Edge Deployment

- [ ] Why Python inference is too slow for production (GIL, interpreter overhead)
- [ ] ONNX Runtime: export your YOLO model, run inference without PyTorch
- [ ] Benchmark: PyTorch vs ONNX Runtime FPS on the same video
- [ ] TensorRT fundamentals: what it does (layer fusion, precision calibration, kernel auto-tuning)
- [ ] FP32 vs FP16 vs INT8 quantization — the speed/accuracy tradeoff
- [ ] Model profiling: where does the time go? (preprocess vs inference vs postprocess vs drawing)
- [ ] Build an optimized inference pipeline that hits maximum FPS on your hardware
- [ ] Docker containerization of your CV pipeline (reproducible deployment)

**Deliverable:** An ONNX-optimized inference pipeline with benchmark comparisons showing speedup over raw PyTorch, packaged in a Docker container.

---

## Week 10: Instance Segmentation

- [ ] Detection vs Segmentation recap: boxes vs pixel-level masks
- [ ] YOLO-Seg: run pre-trained YOLOv11-seg, understand mask output format
- [ ] Mask operations: extracting individual object masks from YOLO-Seg output
- [ ] Combining masks with original image: transparent overlays, background removal
- [ ] Train a custom segmentation model on a drone/aircraft dataset
- [ ] Pixel-level area calculation: estimate real-world object size from mask area + camera calibration
- [ ] Integrate segmentation into your C2 dashboard (colored silhouettes instead of boxes)

**Deliverable:** A segmentation pipeline that draws pixel-perfect drone silhouettes instead of bounding boxes, integrated into the dashboard.

---

## Week 11: Advanced Tracking — Re-ID & Multi-Camera

- [ ] The Re-ID problem: what happens when a tracked object disappears and reappears
- [ ] Appearance descriptors: how DeepSORT uses a feature extractor to re-identify objects
- [ ] Build a simple Re-ID system: extract appearance features, match across track breaks
- [ ] Multi-camera fundamentals: why track IDs from Camera A don't mean anything on Camera B
- [ ] Homography and view mapping: projecting detections from camera view to a 2D floor plan
- [ ] Cross-camera tracking: matching the same object seen by two different cameras
- [ ] Heatmap generation: visualize where objects spend the most time across all cameras

**Deliverable:** A multi-camera tracking demo that maintains consistent object IDs across two camera feeds and projects detections onto a 2D map.

---

## Week 12: Capstone — Multi-Zone Surveillance System

- [ ] Design and build a complete multi-zone surveillance system from scratch
- [ ] Multiple video inputs (phone cameras, video files, RTSP streams)
- [ ] Per-zone threat classification: define zones with different alert levels
- [ ] Dwell-time analysis: alert when an object stays in a zone too long
- [ ] Track trajectory prediction: estimate where an object is heading (linear extrapolation)
- [ ] Event correlation: detect patterns across zones (e.g., object enters Zone A then Zone B within 30 seconds)
- [ ] Export incident reports: timestamped logs with annotated frame snapshots
- [ ] Final portfolio polish: architecture diagram, performance benchmarks, demo video

**Deliverable:** A production-grade multi-zone surveillance system with trajectory prediction, dwell-time alerts, and incident reporting — the centerpiece of your CV portfolio.

---

## Key Concepts to Know for Interviews

These will be woven into the weekly tasks, not studied in isolation:

- **mAP (Mean Average Precision):** How detection accuracy is measured
- **IoU (Intersection over Union):** The foundation of mAP calculation and NMS
- **NMS (Non-Maximum Suppression):** How overlapping detections are filtered
- **Loss:** What the model optimizes during training
- **Overfitting:** When the model memorizes training data instead of learning patterns
- **Data Leakage:** When test data contaminates training, giving fake-good metrics
- **ONNX/TensorRT:** How Python prototypes become C++ production systems
- **FPS vs Accuracy tradeoff:** Nano vs Large models, when to use which
- **Quantization:** FP32 → FP16 → INT8 and the speed/accuracy implications
- **Re-ID:** How tracking systems handle object disappearance and reappearance
- **Homography:** Mapping between camera views and real-world coordinates
- **Edge vs Cloud inference:** Latency, bandwidth, and privacy tradeoffs
