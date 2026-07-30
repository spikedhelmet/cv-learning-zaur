# AI Context - Computer Vision Learning for Defense

## Role Context & Background
The user is a frontend developer (experienced in React Native, TypeScript, PHP, and Git, working across macOS and Windows) transitioning into an applied Computer Vision engineering role for a defense company. The target role focuses on counter-drone systems (detecting and tracking small targets in real-time).

## The 2-Month Roadmap
We have a strict 2-month timeline to build a portfolio project (a full-stack Defense Command & Control Dashboard with a CV backend) and prepare for interviews. We are skipping heavy math and building neural networks from scratch in C, and instead focusing strictly on applied, hands-on engineering.

- **Month 1 (Foundations):** Python, NumPy, and OpenCV. Learning how to manipulate images as 3D matrices, read video streams, crop regions, and draw bounding boxes.
- **Month 2 (Detection & Tracking):** Using Roboflow for drone datasets and augmentations. Training and fine-tuning YOLO (v8/11) models. Implementing multi-object tracking (like ByteTrack) to keep persistent IDs on moving targets.
- **Core Concepts to master practically:** mAP, Loss, Overfitting, and Data Leakage.
- **Production Awareness:** Prototype and train in Python, but keep in mind that production defense systems run on C++ (via ONNX/TensorRT) for edge hardware. Explain system architecture with this context.

## Current Progress (Day 1 & Day 2)
- We have set up a Python virtual environment (`cv-env`).
- Installed baseline libraries (`numpy`, `opencv-python`, `matplotlib`).
- Wrote and tested `day1_demo.py` which demonstrates basic image creation (NumPy array as a 500x500 canvas) and drawing a bounding box using OpenCV.
- Mapped basic Python/NumPy concepts (like strictly typed matrices and BGR color space) to TypeScript and HTML5 Canvas paradigms.
- Wrote `day2_image.py` which successfully loads a real image using `cv2.imread()`, gets its dimensions via NumPy `.shape`, and draws a simulated bounding box.
- Successfully wrote `day2_webcam.py` to capture live video from the webcam using `cv2.VideoCapture(0)`.
- Understood OpenCV idioms like `waitKey` masking and `cap.read()` destructuring.
- Successfully implemented live real-time YOLO preprocessing (Grayscale, Resize 640x640, and Array slicing/cropping).

## Next Steps
- Learn to save/write modified video streams to a file on disk (crucial for recording drone flights and saving YOLO detection outputs).

*Note for AI: When this repository is cloned on a new computer, read this file to instantly resume the 2-month roadmap and maintain the user's specific learning context.*
