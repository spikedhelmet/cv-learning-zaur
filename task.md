# Month 2: Object Detection with YOLO and Roboflow

Congratulations! You have officially completed the **Month 1 Foundations**. You now understand how to read video streams, manipulate them as matrices in real-time, and save the output. That is the exact pipeline needed for applied Computer Vision.

Now we enter the exciting part: **Detection & Tracking.**

In defense systems, you don't just look at pixels—you need the computer to draw a box around a drone and tell you "Drone detected with 95% confidence at coordinates X, Y".

To do this, we use **YOLO (You Only Look Once)**. It's the industry standard for real-time object detection because it's incredibly fast.

### Phase 1: Understanding the YOLO Pipeline

Before we train our own model on custom drone data, we are going to use a pre-trained YOLO model to detect everyday objects (people, laptops, cell phones, etc.) to understand the workflow.

For this phase, we need to install the `ultralytics` package, which is the official Python library for modern YOLO models (v8 and v11).

### Your First Task in Month 2:

**1. Install Ultralytics:**
Open your terminal (make sure your `cv-env` is active!) and run:
```bash
pip install ultralytics
```

**2. Create a new file:**
Create a file called `day3_yolo_intro.py`.

**3. Run Inference on an Image:**
Write a script that downloads and runs YOLO on a default image. Ultralytics makes this surprisingly easy. Here is the code to get you started:

```python
from ultralytics import YOLO
import cv2

# Load a pre-trained YOLO11 Nano model (it will auto-download the weights file 'yolo11n.pt')
model = YOLO('yolo11n.pt')

# Run inference on an image. (You can use any image you have, or pass a URL)
# The model will download this sample image from the internet if you don't have it locally.
results = model('https://ultralytics.com/images/zidane.jpg')

# YOLO returns a list of Results objects. We just take the first one.
result = results[0]

# result.plot() automatically draws the bounding boxes and labels onto the image!
annotated_frame = result.plot()

# Show it using your OpenCV skills!
cv2.imshow("YOLO Inference", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Run this script and see what happens! It might take a few seconds the very first time because it has to download the tiny 6MB `yolo11n.pt` model weights file.

Let me know when you get the image to pop up with boxes drawn around the people!
