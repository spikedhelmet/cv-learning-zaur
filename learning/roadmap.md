# Applied Computer Vision Roadmap (v2)

**Goal:** get hired as a CV/ML engineer, ideally at a defense company working on counter-drone systems.

**Start date:** 2026-07-29. **v2 written:** 2026-10-22, at the end of Week 9 Task 1, Step 6.

---

## Why v2

Weeks 1-7 taught you to use the Ultralytics API: `model.train()`, `model.track()`, `results[0].boxes`. That is useful, but CV/ML interviews go below the API. Expect questions like "write NMS", "write a training loop", "why does this loss look like that", "what do the camera intrinsics mean".

Week 9 Task 1 showed the gap. Writing YOLO postprocessing by hand was hard mostly because of NumPy (shapes, axes, masks, indexing), not because of ONNX. The task doc also handed you the code for every step, so you understood it while reading but could not reproduce it afterwards.

v2 does two things:

1. **Teaches foundations before features:** NumPy, PyTorch, the math intuition behind them, model internals, tracking internals, camera geometry, and some C++.
2. **Changes how each unit works** (next section).

---

## How every unit works

1. **Short concept section.** No solution code.
2. **A spec plus expected outputs.** Check code is provided; solution code is not.
3. **You write it in a blank file.** Docs are allowed for API signatures (`np.argsort`, `cv2.resize`), but not for looking up solutions.
4. **Hints come in tiers**, in collapsed blocks inside the drill. Open them one at a time, and only after you have been stuck for about 10 minutes.
5. **Review.** Send the code to Claude for review, then answer the interview questions at the bottom of the drill (out loud or in writing).
6. **Rep 2:** at least one day later, delete the file and write it again from a blank file, with no notes and no hints.

**A unit is done only after Rep 2 passes.** Rep 1 shows you understood it. Rep 2 shows you can do it, and doing it is what interviews test.

Each drill below has two checkboxes: `Rep 1` and `Rep 2`.

---

## Completed: v1 Weeks 1-9 (summary)

| Week | Topic | Level reached |
| --- | --- | --- |
| 1 | OpenCV basics: color spaces, thresholding, morphology, histograms/CLAHE | Used with guidance |
| 2 | Background subtraction, contours, Canny, Lucas-Kanade optical flow, ROI | Used with guidance |
| 3 | Pretrained YOLO, reading its outputs, FPS | API level |
| 4 | Datasets, annotation formats, splits, leakage, augmentation (Roboflow) | Conceptual |
| 5 | Fine-tuning YOLO, loss/mAP curves, overfitting, ONNX export | API level (`model.train()`) |
| 6 | ByteTrack through Ultralytics, IDs, tripwire counting | API level (`model.track()`) |
| 7 | FastAPI backend: WebSocket, MJPEG, SQLite events, zone logic | Solid (closest to your day job) |
| 8 | Frontend dashboard | Skipped (not CV learning) |
| 9 | ONNX Runtime detector with hand-written pre/postprocessing, verified against PyTorch (Steps 1-6 of `week9/week9_task1_onnx_benchmark.md`) | Done, but not reproducible yet. Phase A fixes that. |

Old Weeks 10-12 (segmentation, Re-ID/multi-camera, capstone) are not dropped. They are spread over the phases below.

---

## Phase A: NumPy & detection primitives (~2 weeks)

Folder: `learning/phaseA/`. Each drill is one `.md` file with its own `.py` file next to it.

**Why first:** IoU and NMS from scratch are some of the most common CV coding interview questions. By the end of this phase you will rebuild the Week 9 detector from a blank file.

| Drill | Topic | Rep 1 | Rep 2 |
| --- | --- | --- | --- |
| A1 | IoU of two boxes (pure Python) | [ ] | [ ] |
| A2 | NumPy fluency: shapes, axes, masks, indexing, broadcasting | [ ] | [ ] |
| A3 | Vectorized IoU: one-to-many and N x M matrix | [ ] | [ ] |
| A4 | Box format conversions and clipping | [ ] | [ ] |
| A5 | NMS from scratch, then per-class NMS | [ ] | [ ] |
| A6 | Letterbox and mapping boxes both ways | [ ] | [ ] |
| A7 | Decode the raw YOLO output tensor | [ ] | [ ] |
| A8 | **Exit test:** rebuild the ONNX detector from a blank file in 45 minutes | [ ] | — |

---

## Phase B: PyTorch fundamentals (~3 weeks)

**Why:** right now "I trained a model" means "I called `model.train()`." An interviewer will ask what happens inside it.

- B1 Tensors vs NumPy arrays: dtype, device (`mps`), shapes, `view` vs `reshape`
- B2 Autograd: `requires_grad`, `.backward()`, `.grad`, why `zero_grad` exists
- B3 Linear regression with autograd and a hand-written gradient descent loop, no `nn.Module`
- B4 `nn.Module`, `nn.Linear`, an optimizer, and a full training loop (train/val split, `model.eval()`, `torch.no_grad()`)
- B5 `Dataset` and `DataLoader` for your own images: drone crops vs background crops cut from the Week 4 dataset
- B6 A small CNN trained from scratch on those crops. Overfit it on purpose, then fix it with augmentation, weight decay, and early stopping.
- B7 Loss functions: BCE vs cross-entropy, logits vs probabilities, and why `BCEWithLogitsLoss` exists
- B8 Saving and loading weights, `state_dict`, and exporting your own model to ONNX
- B9 **Exit test:** write the training loop from B4-B6 from a blank file

## Phase C: Math intuition (alongside Phase B)

Enough to answer "why" questions; this is not a math degree. Each item is a NumPy drill.

- C1 Convolution by hand in NumPy: kernel, stride, padding, and the output-size formula
- C2 Gradient descent on a 2-variable function, plus the effect of the learning rate
- C3 The chain rule on a 2-layer network: compute the gradients by hand, then check them against autograd
- C4 Sigmoid, softmax, and cross-entropy written from scratch, including numerical stability (log-sum-exp)
- C5 Precision, recall, the PR curve, AP, and mAP@0.5 computed from scratch on your model's predictions
- C6 BatchNorm, dropout, and receptive field (conceptual, plus a small demo)

## Phase D: Deployment & optimization (~1.5 weeks)

Resumes v1 Week 9.

- D1 Benchmark PyTorch CPU/MPS vs ORT CPU/CoreML (Steps 7-9 of `week9/week9_task1_onnx_benchmark.md`)
- D2 FP16 and INT8 quantization: speed vs accuracy, measured with C5's mAP code
- D3 TensorRT concepts: layer fusion, calibration, kernel auto-tuning (on paper; no NVIDIA GPU here)
- D4 Docker image for the inference pipeline

## Phase E: Detector internals & data (~2 weeks)

- E1 YOLO architecture: backbone, neck, head, anchor-free head, strides, and why 8400 candidates
- E2 YOLO loss components (box/IoU loss, classification, DFL) and what each training curve means
- E3 Error analysis: sort false positives and false negatives into types, and find what the model actually gets wrong
- E4 Small-object detection (drones are tiny at range): input resolution, tiling/SAHI, and the trade-offs
- E5 Collect and label your own small drone/bird test set, and evaluate on it honestly

## Phase F: Tracking internals (~2 weeks)

**Why:** a counter-drone system is a tracking system. "I used ByteTrack" isn't enough.

- F1 Kalman filter from scratch (1D, then a 2D constant-velocity box)
- F2 Hungarian matching (`scipy.optimize.linear_sum_assignment`) on an IoU cost matrix (reuses A3)
- F3 SORT from scratch: predict, match, update, track birth and death
- F4 Compare against ByteTrack on your video: ID switches and fragmentation
- F5 Re-ID: appearance embeddings and matching across track breaks (old Week 11)
- F6 Trajectory prediction and dwell time (old Week 12)

## Phase G: Camera geometry (~1.5 weeks)

- G1 Pinhole model, intrinsics matrix, and projecting 3D points to pixels by hand
- G2 Chessboard calibration of your phone camera, plus lens distortion and undistortion
- G3 Homography: image to ground plane / map (old Week 11)
- G4 Pixel size to distance estimation: how far away a drone of known size is
- G5 Multi-camera: shared map coordinates and cross-camera matching (old Week 11)

## Phase H: C++ inference (~2 weeks)

- H1 Modern C++ basics needed for CV: build with CMake, `std::vector`, references, RAII
- H2 OpenCV in C++: read video, letterbox, draw
- H3 ONNX Runtime C++ API: port your Phase A detector (letterbox, decode, NMS) to C++
- H4 Benchmark the C++ vs Python pipelines

## Phase I: Capstone & job prep

- I1 Capstone: multi-camera or multi-zone system built on your own detector, tracker, and geometry code (old Week 12)
- I2 Portfolio: README, architecture diagram, benchmark table, demo video
- I3 Interview prep: re-run every exit test, mock coding rounds (IoU, NMS, conv output size, training loop, Kalman update), and system design ("design a counter-drone detection pipeline")
- Optional: segmentation (old Week 10), if target job postings ask for it

---

## When to start applying

Start applying after **Phases A-C and F** are done (all exit tests passed, Rep 2 included) and at least one of D/E is done. At that point you can:

- write IoU, NMS, and a training loop from a blank file,
- explain the YOLO head, its loss, and mAP from first principles,
- build a tracker from a Kalman filter and Hungarian matching,
- back all of it with your own measured benchmarks.

That is a credible junior/mid CV engineer profile. Phases G-H make it stronger for defense roles specifically (sensor geometry, C++ on edge devices).

---

## Interview concepts checklist

Each one is covered by the drill named after it.

| Concept | Where |
| --- | --- |
| IoU | A1, A3 |
| NMS (class-agnostic vs per-class) | A5 |
| Letterbox / preprocessing | A6 |
| mAP, precision/recall | C5 |
| Loss functions, logits | B7, C4 |
| Backprop / chain rule | B2, C3 |
| Convolution output size | C1 |
| Overfitting, regularization | B6 |
| Data leakage | v1 Week 4, E5 |
| ONNX / TensorRT | v1 Week 9, D1-D3 |
| Quantization | D2 |
| Anchor-free detection head | E1 |
| Small-object detection | E4 |
| Kalman filter, Hungarian matching | F1-F3 |
| Re-ID | F5 |
| Intrinsics, homography | G1-G3 |
| Edge vs cloud inference | D, H |
