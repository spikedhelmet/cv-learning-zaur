# AI Context - Computer Vision Learning for Defense

## Role Context & Background
The user is a frontend developer (experienced in React Native, TypeScript, PHP, and Git, working across macOS and Windows) transitioning into an applied Computer Vision engineering role for a defense company. The target role focuses on counter-drone systems (detecting and tracking small targets in real-time).

## Learning Style & Instructions for AI
- Do not over-praise or "glaze." Be direct.
- We have a full 8 weeks. Do not rush or skip content.
- The detailed week-by-week plan lives in `roadmap.md`. Reference it before creating new tasks.
- Each new task should be a new file (e.g., `week1_task3_thresholding.md`), not an overwrite of previous ones.
- Map new concepts to TypeScript/Web equivalents when it helps understanding.
- Focus on applied, hands-on engineering. Skip heavy math theory.

## The 8-Week Roadmap
See `roadmap.md` for the full breakdown. Summary:
- **Weeks 1-2:** OpenCV foundations, classical CV techniques, motion detection
- **Weeks 3-4:** Deep learning intro, YOLO inference, datasets & Roboflow
- **Weeks 5-6:** Custom YOLO training, multi-object tracking (ByteTrack)
- **Weeks 7-8:** Defense C2 Dashboard (FastAPI backend + React frontend), portfolio polish

**Core Concepts** (woven into tasks, not studied in isolation): mAP, Loss, Overfitting, Data Leakage.
**Production Awareness:** Prototype in Python, production runs on C++ (ONNX/TensorRT) for edge hardware.

## Environment
- Python virtual environment: `cv-env`
- Libraries installed: `numpy`, `opencv-python`, `matplotlib`
- No physical webcam on Mac — using Android phone with IP Webcam app over Wi-Fi
- IP Webcam URL format: `http://<phone-ip>:8080/video`

## Current Progress
- **Week 1, Days 1-2 complete.** See `roadmap.md` for checked-off items.
- Files written: `day1_demo.py`, `day2_image.py`, `day2_webcam.py`
- Currently on: Week 1, remaining tasks (color spaces, thresholding, morphological ops, histograms)

*Note for AI: When this repository is cloned on a new computer, read this file and `roadmap.md` to instantly resume.*
