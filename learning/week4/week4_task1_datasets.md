# Week 4 — Task 1: Datasets, Annotations & the YOLO Training Format

---

## Goal

Understand how object detection datasets are structured, what annotation formats exist, how to evaluate dataset quality, and how to set up the exact file structure YOLO expects for training. This is the foundation for Week 5, where you'll train your own model — and bad data is the #1 reason models fail in production.

---

## Concept: Why Data Matters More Than Code

In classical CV (Weeks 1-2), you wrote the rules. The quality of your pipeline depended on how clever your code was.

In deep learning, the code is almost identical across projects. What changes is the **data**. Two engineers using the exact same YOLO architecture with the exact same hyperparameters will get wildly different results if one has a clean, diverse dataset and the other has a noisy, biased one.

The industry saying is: **"Garbage in, garbage out."** If you train on blurry images, your model learns to detect blurry things. If all your drone images are taken at noon with blue sky, your model will fail at dusk.

### Defense Context

In counter-drone systems, dataset quality directly determines whether the system works in the field:
- A model trained only on DJI Mavics will miss fixed-wing drones.
- A model trained only on clear-sky backgrounds will fail when clouds are present.
- A model trained on high-resolution images will fail on low-resolution thermal cameras.
- If test images leak into the training set, your metrics look great in the lab but the system fails on deployment. This is called **data leakage** and it's a career-ending mistake in defense contracts.

### Web/TS Analogy

Think of a dataset like a test suite. If your unit tests only cover the happy path, your app crashes in production on edge cases. A good dataset is like a comprehensive test suite: it covers normal cases, edge cases, boundary conditions, and adversarial inputs. And just like you'd never put your test fixtures into your production build, you must never let test data leak into training data.

---

## Technical Mechanics

### 1. What Makes a Good Detection Dataset

A detection dataset is a collection of images paired with **annotation files** that describe where objects are in each image (bounding boxes + class labels).

**Quality criteria:**

| Criterion | What it means | Bad example | Good example |
|-----------|--------------|-------------|--------------|
| **Diversity** | Variety in backgrounds, lighting, angles, distances | All images taken in the same room | Images from forests, cities, beaches, day/night |
| **Balance** | Roughly equal number of examples per class | 5000 "car" images, 12 "truck" images | 2000 cars, 1800 trucks |
| **Scale variety** | Objects at different sizes in the frame | All drones filling 50% of the image | Drones at 10px, 50px, 200px, 500px wide |
| **Negative samples** | Images with NO target objects | Every image has a drone | 20% of images are just sky/buildings (no drone) |
| **Annotation quality** | Tight, consistent bounding boxes | Boxes that loosely surround objects with lots of padding | Boxes that tightly wrap the visible object boundary |
| **Edge cases** | Unusual but real-world scenarios | Only solo drones | Overlapping drones, partially occluded, motion blur |

**Common dataset sizes for custom training:**

| Use case | Typical size | Notes |
|----------|-------------|-------|
| Quick prototype / proof of concept | 100-500 images | Fine-tune a pre-trained model. Expect mediocre results. |
| Solid working model | 1,000-5,000 images | The sweet spot for most custom detection tasks. |
| Production-grade model | 10,000+ images | What defense companies aim for. Requires serious annotation effort. |

### 2. Annotation Formats

Every annotation format answers the same question differently: *"What objects are in this image, where are they, and what class are they?"*

#### YOLO TXT Format (what we'll use)

One `.txt` file per image, same filename. Each line is one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0.0 to 1.0), relative to image dimensions.

Example — an image `frame_001.jpg` (640×480 pixels) with a drone and a bird:
```
0 0.45 0.30 0.12 0.08
1 0.78 0.62 0.05 0.04
```

Breaking down the first line:
- `0` → class ID (maps to "drone" via `data.yaml`)
- `0.45` → x_center = 288px / 640px (the horizontal center of the bounding box)
- `0.30` → y_center = 144px / 480px (the vertical center)
- `0.12` → width = 76.8px / 640px (the box width)
- `0.08` → height = 38.4px / 480px (the box height)

**Why normalized?** Because images come in different resolutions. Normalized coordinates work regardless of whether the image is 640×480 or 1920×1080. The model doesn't care about pixel counts — it cares about relative positions.

**Why center-based?** YOLO internally predicts box centers and dimensions, so the training format matches the prediction format. No conversion needed.

#### COCO JSON Format

A single `.json` file for the entire dataset. Used by the COCO benchmark and many academic papers.

```json
{
  "images": [
    {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [249.6, 124.8, 76.8, 38.4]
    }
  ],
  "categories": [
    {"id": 0, "name": "drone"},
    {"id": 1, "name": "bird"}
  ]
}
```

Key difference: COCO uses **absolute pixel coordinates** and `[x_top_left, y_top_left, width, height]` — not center-based, not normalized.

#### Pascal VOC XML Format

One `.xml` file per image. The oldest format, used by the original Pascal VOC challenge (2005-2012).

```xml
<annotation>
  <filename>frame_001.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object>
    <name>drone</name>
    <bndbox>
      <xmin>211</xmin><ymin>106</ymin>
      <xmax>288</xmax><ymax>144</ymax>
    </bndbox>
  </object>
</annotation>
```

Key difference: Uses absolute pixel coordinates with `[xmin, ymin, xmax, ymax]` corners.

#### Format Comparison

| Format | File type | Coordinates | One file per... | Used by |
|--------|-----------|-------------|-----------------|---------|
| YOLO TXT | `.txt` | Normalized center `(cx, cy, w, h)` | Image | Ultralytics, Darknet |
| COCO JSON | `.json` | Absolute top-left `(x, y, w, h)` | Entire dataset | COCO benchmark, Detectron2 |
| Pascal VOC | `.xml` | Absolute corners `(xmin, ymin, xmax, ymax)` | Image | Older frameworks, some APIs |

**For this course, we use YOLO TXT.** Roboflow can convert between all three.

### 3. Dataset Splits: Train / Val / Test

You **never** evaluate a model on data it was trained on. That's like a student grading their own homework.

| Split | Purpose | Typical % | When it's used |
|-------|---------|-----------|----------------|
| **Train** | The model learns from these images | 70% | During training |
| **Val** (Validation) | Used to tune hyperparameters and monitor overfitting during training | 15% | During training (after each epoch) |
| **Test** | Final, untouched evaluation. The model never sees these until the very end. | 15% | After training is complete |

**Data Leakage** happens when test/val images end up in the training set. Your metrics will look artificially high, but the model hasn't actually learned to generalize. In defense, this means a system that passes all lab tests but fails in the field.

Common leakage mistakes:
- Augmenting an image and putting the original in train, the augmented copy in val.
- Extracting frames from the same video and splitting them across train/val/test (consecutive frames look nearly identical).
- Shuffling after augmentation instead of before.

### 4. The YOLO `data.yaml` Config

This file tells Ultralytics where your data lives and what the classes are:

```yaml
# data.yaml
path: /absolute/path/to/dataset   # Root directory of the dataset
train: images/train               # Relative to 'path'
val: images/val                   # Relative to 'path'
test: images/test                 # Relative to 'path' (optional)

nc: 2                             # Number of classes
names: ['drone', 'bird']          # Class names, order matches class IDs (0='drone', 1='bird')
```

**Directory structure YOLO expects:**

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── frame_100.jpg
│   │   └── ...
│   └── test/
│       ├── frame_200.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── frame_001.txt    ← Same filename as the image, different extension
    │   ├── frame_002.txt
    │   └── ...
    ├── val/
    │   ├── frame_100.txt
    │   └── ...
    └── test/
        ├── frame_200.txt
        └── ...
```

**Critical rule:** Each label `.txt` file must have the **exact same name** as its corresponding image (just `.txt` instead of `.jpg`). If the image is `frame_001.jpg`, the label must be `frame_001.txt`. YOLO matches them by filename.

---

## Step-by-Step Task

This week is less about writing Python scripts and more about understanding data. But there's still hands-on work.

### 1. Create the directory structure

Inside your project, create the following empty directory tree. Since you are using Windows PowerShell, you can run this command to create all the folders at once:

```powershell
"train","val","test" | ForEach-Object { New-Item -ItemType Directory -Force -Path "month1\week4\drone_dataset\images\$_"; New-Item -ItemType Directory -Force -Path "month1\week4\drone_dataset\labels\$_" }
```

This gives you the exact skeleton YOLO expects.

### 2. Write a `data.yaml` file

Create `month1/week4/drone_dataset/data.yaml` by hand. Define:
- `path`: the absolute path to your `drone_dataset/` directory
- `train`, `val`, `test`: the relative paths to image subdirectories
- `nc`: 1 (we'll start with just "drone")
- `names`: `['drone']`

### 3. Create a manual annotation by hand

Take a screenshot of anything (or download any image from the internet). Place it in `images/train/`. Then create a corresponding `.txt` file in `labels/train/` with the same filename.

Manually calculate the normalized YOLO coordinates for one object in the image:
1. Open the image, note its pixel dimensions (width × height).
2. Identify an object. Estimate or measure the bounding box corners in pixels: `(xmin, ymin, xmax, ymax)`.
3. Convert to YOLO format:
   - `x_center = (xmin + xmax) / 2 / image_width`
   - `y_center = (ymin + ymax) / 2 / image_height`
   - `box_width = (xmax - xmin) / image_width`
   - `box_height = (ymax - ymin) / image_height`
4. Write the line: `0 <x_center> <y_center> <box_width> <box_height>`

This exercise is painful on purpose. It makes you viscerally understand why annotation tools exist and exactly what the numbers in a label file represent.

### 4. Write a Python script to verify your annotation

Create `month2/week4/verify_annotation.py`:
- Load the image with OpenCV.
- Read the corresponding `.txt` label file.
- Parse each line, convert the normalized coordinates back to pixel coordinates.
- Draw the bounding box on the image using `cv2.rectangle`.
- Display the result.

If your box tightly wraps the object you annotated, your manual annotation is correct.

### 5. Explore an existing dataset on Roboflow

Go to [Roboflow Universe](https://universe.roboflow.com/) and search for "drone detection" datasets. Browse 2-3 datasets and observe:
- How many images do they have?
- How many classes?
- What does the class distribution look like (balanced or skewed)?
- What format options does Roboflow offer for download?

Don't download anything yet — just explore and get a feel for what's available. We'll download and set up a real dataset in the next task.

---

## Checkpoint Questions

1. Why are YOLO annotation coordinates normalized (0.0-1.0) instead of using absolute pixel values?
2. If you extract 100 frames from a 10-second video clip and randomly split them 70/15/15 into train/val/test, is this a valid split? Why or why not?
3. Your dataset has 5000 images of drones and 50 images of birds. What problem does this cause, and how would you address it?
4. What is the difference between the validation set and the test set? Why do you need both?

---

## Challenge (No Guidance)

**Annotation Format Converter**

Create `month2/week4/format_converter.py`:

1. Write a function that takes a YOLO-format annotation line (`"0 0.45 0.30 0.12 0.08"`) and an image size tuple `(width, height)`, and converts it to:
   - COCO format: `{"category_id": 0, "bbox": [x_topleft, y_topleft, width, height]}` (absolute pixels)
   - Pascal VOC format: `{"name": "drone", "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}` (absolute pixels)
2. Write the reverse: a function that takes COCO-format bbox and image size, and returns a YOLO-format string.
3. Test both functions on your manual annotation from Step 3. Verify the round-trip: YOLO → COCO → YOLO should give you back the original numbers (within floating-point precision).

This is pure coordinate math — no ML, no OpenCV magic. But this exact conversion code exists in every production CV pipeline.

---

## Supplemental Reading

**For interviews:**
- **"What is data leakage?"** — Be ready to explain this with a concrete example. The video-frame-splitting scenario above is the most common one interviewers ask about. The key insight: leakage doesn't cause an error — your code runs fine, your metrics look great, but your model doesn't actually generalize.
- **"How do you handle class imbalance?"** — Common strategies: oversample the minority class, undersample the majority class, use weighted loss functions, or generate synthetic examples via augmentation. Know at least two approaches.
- **"What's the difference between COCO and YOLO format?"** — Coordinate system (absolute vs normalized) and file structure (one JSON vs one txt per image). Simple question but it shows you've actually worked with data.

**For production context:**
- **Active learning** — In defense, you can't annotate everything. Active learning is a strategy where the model flags images it's uncertain about, and human annotators only label those. This reduces annotation cost by 60-80%. Companies like Scale AI and Labelbox provide this as a service.
- **Annotation quality assurance** — Production datasets have multiple annotators label the same image, then measure inter-annotator agreement (IoU between their boxes). If annotators disagree on where the object boundary is, the annotation instructions need to be more specific.

**External resources:**
- Roboflow Universe (browse datasets): https://universe.roboflow.com/
  - **Prerequisite:** None. **Takeaway:** See how real-world detection datasets are structured, what class distributions look like, and what download format options exist. Spend 10 minutes exploring drone/UAV datasets.
- Ultralytics docs — Datasets guide: https://docs.ultralytics.com/datasets/
  - **Prerequisite:** Understanding of YOLO format from this task. **Takeaway:** Reference for the `data.yaml` format and directory structure. Bookmark for Week 5.
