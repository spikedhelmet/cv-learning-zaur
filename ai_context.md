# AI Context - Computer Vision Learning for Defense

## Role Context & Background
The user is a frontend developer (experienced in React Native, TypeScript, PHP, and Git, working across macOS and Windows) transitioning into an applied Computer Vision engineering role for a defense company. The target role focuses on counter-drone systems (detecting and tracking small targets in real-time).

## Learning Style & Instructions for AI
- Do not over-praise or "glaze." Be direct.
- **CRITICAL**: This is a learning project. Do NOT directly edit the user's Python scripts to solve problems. Provide explanations, snippets, or guidance in chat, and let the user implement the code.
- We have a full 8 weeks. Do not rush or skip content.
- The detailed week-by-week plan lives in `roadmap.md`. Reference it before creating new tasks.
- Each new task should be a new file (e.g., `week1_task3_thresholding.md`), not an overwrite of previous ones.
- Map new concepts to TypeScript/Web equivalents when it helps understanding.
- Focus on applied, hands-on engineering. Skip heavy math theory.
- **Include occasional unguided challenges** — problems where the user must figure out the implementation without step-by-step code. This builds real problem-solving skill. Especially do it at the end of weeks. Occasional quizzes can also help.
- **Provide supplemental theory references** where useful (interview prep, deeper understanding). Links, book chapters, papers, etc. Don't force it, but don't skip it when it matters.
- **Supplemental reading must include context:** For every external resource (video, article, book chapter), add a one-line note stating: (1) what prerequisite knowledge is needed, (2) what specific takeaway the learner should get from it, and (3) whether it's relevant now or better saved for a later week. Don't recommend resources that require knowledge the learner hasn't built yet.
- **CRITICAL: Explain every function and type used in tasks.** When introducing any function, method, or constructor, always explain its full signature: what every argument does, what type it expects, and what it returns. Never drop a code snippet without explaining what each piece does. This applies to NumPy, OpenCV, and Python built-ins alike. Use TypeScript/JS analogies where they map cleanly.

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
- **Week 1 complete.** See `roadmap.md` for checked-off items.
- Files written: `day1_demo.py`, `day2_image.py`, `day2_webcam.py`, `week1_color_spaces.py`, `week1_morphology.py`, `week1_challenge_isolator.py`, `week1_histograms.py`
- Completed: color spaces (HSV), thresholding (binary, Otsu, adaptive), morphological operations (erosion, dilation, opening, closing), histograms (equalization, CLAHE)
- Completed the Week 1 unguided challenge: built an object isolator using HSV color masking + morphological opening + contour detection + area filtering
- **Week 2 near-complete.** Background subtraction, contour detection, motion detector, Canny edges, optical flow (Lucas-Kanade), ROI selection & masking all done.
- Completed Week 2 challenges: optical flow direction indicator, ROI motion detection zone
- Currently finishing: **Week 2 — wrapping up, ready to move to Week 3**

*Note for AI: When this repository is cloned on a new computer, read this file and `roadmap.md` to instantly resume.*
