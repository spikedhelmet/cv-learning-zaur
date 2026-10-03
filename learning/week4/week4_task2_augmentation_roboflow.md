# Week 4 — Task 2: Data Augmentation & Building a Real Drone Dataset with Roboflow

---

## Goal

Understand **data augmentation** (artificially expanding your dataset), set up a **Roboflow** account, download a real drone detection dataset, and prepare it in the YOLO format ready for training. By the end, you'll have a properly structured dataset in `drone_dataset/` with hundreds of annotated images.

---

## Concept: Data Augmentation

Training a model requires thousands of images. Collecting and annotating that many by hand is expensive and slow. **Data augmentation** creates new training images by applying random transformations to your existing images.

### Common Augmentations

| Augmentation            | What it does                          | Why it helps                                                                                                |
| ----------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Horizontal Flip**     | Mirrors the image left-to-right       | A drone flying left looks different from one flying right. Flipping doubles your data for free.             |
| **Rotation (±15°)**     | Slightly tilts the image              | Camera angles vary in the field. The model needs to handle slight rotations.                                |
| **Brightness/Contrast** | Randomly adjusts lighting             | Drones appear in dawn, noon, dusk, overcast. The model must handle all conditions.                          |
| **Noise**               | Adds random pixel noise               | Simulates low-quality cameras, sensor noise, or compression artifacts.                                      |
| **Crop**                | Randomly crops a portion of the image | Forces the model to detect partially visible objects (drone half-off-screen).                               |
| **Mosaic**              | Combines 4 images into one            | A YOLO-specific augmentation. Forces the model to detect multiple objects at different scales in one image. |

### Critical Rule: Augment ONLY the Training Set

Never augment your `val` or `test` sets. Those must remain pristine, unmodified images that represent real-world conditions. Augmenting evaluation data would inflate your metrics artificially.

### Defense Context

In counter-drone systems, you encounter drones at different altitudes (tiny vs. large in frame), different times of day (lighting), different weather (rain, fog, glare), and different backgrounds (sky, trees, buildings). Augmentation simulates these variations without needing to physically fly a drone in every possible condition.

---

## Technical Mechanics

### What is Roboflow?

Roboflow is a platform for managing CV datasets. It handles:

1. **Uploading** images and annotations
2. **Annotating** (labeling) images with bounding boxes in a web UI
3. **Augmenting** your dataset with configurable transformations
4. **Exporting** in any format (YOLO, COCO, Pascal VOC, etc.)
5. **Versioning** your dataset (like Git for datasets)

For this course, we'll use Roboflow to **find and download an existing drone dataset** that someone else has already annotated. Later (Week 5), you can use it to annotate your own custom data.

### Roboflow Universe

Roboflow Universe (https://universe.roboflow.com) is a public repository of 200,000+ datasets. You can search for "drone detection", preview the images and labels, and download in YOLO format with one click.

### Augmentation in Roboflow vs. in Code

- **Roboflow augmentation:** Applied when you create a "Version" of your dataset. Roboflow generates the augmented images and includes them in the download. Simple, no code required.
- **Code-based augmentation (Albumentations library):** Applied on-the-fly during training. Each epoch, the model sees slightly different versions of the same images. More flexible, but requires Python code. YOLO's built-in training pipeline (Ultralytics) already applies mosaic, flips, and color jitter automatically.

For now, we'll use Roboflow's built-in augmentation. You'll encounter code-based augmentation in Week 5 when you start training.

---

## Step-by-Step Task

### 1. Create a Roboflow Account

1. Go to https://roboflow.com and sign up (free tier is sufficient).
2. Create a new workspace (name it whatever you want, e.g., "CV Learning").

### 2. Find a Drone Detection Dataset on Roboflow Universe

1. Go to https://universe.roboflow.com
2. Search for **"drone detection"**.
3. Browse the results. Look for a dataset that has:
   - At least **500+ images** (more is better)
   - **Bounding box annotations** (not segmentation masks)
   - A good mix of backgrounds (sky, urban, rural)
   - Recent uploads (more likely to be well-labeled)
4. Some recommended datasets to look for:
   - "Drone Detection" by various users
   - "Anti-UAV" datasets
   - "Drone vs Bird" datasets (these are great because they include a confuser class)

### 3. Explore the Dataset

Before downloading, explore the dataset on Roboflow:

- **Browse images:** Click through 20-30 images. Are the bounding boxes tight and accurate? Are there unlabeled objects?
- **Check class distribution:** How many images per class? Is it balanced?
- **Check image quality:** Are images diverse (different backgrounds, lighting, drone sizes)?

### 4. Download in YOLO Format

1. Click **"Download Dataset"** (or "Download this Dataset").
2. Select **YOLOv8** (or YOLOv11) as the format.
3. Choose **"download zip to computer"**.
4. Select the version that includes augmentations (or create a new version with augmentations like horizontal flip, brightness ±15%, and rotation ±10°).
5. Unzip the downloaded file.

### 5. Organize into Your Project Structure

The Roboflow download will come with its own `train/`, `valid/`, `test/` folders and a `data.yaml`. Move or copy the contents into your existing project structure:

```
month1/week4/drone_dataset/
├── data.yaml           ← Update this with correct paths
├── images/
│   ├── train/          ← Copy Roboflow's train images here
│   ├── val/            ← Copy Roboflow's valid images here
│   └── test/           ← Copy Roboflow's test images here
└── labels/
    ├── train/          ← Copy Roboflow's train labels here
    ├── val/            ← Copy Roboflow's valid labels here
    └── test/           ← Copy Roboflow's test labels here
```

### 6. Update `data.yaml`

Edit your `data.yaml` to reflect the actual dataset:

- Update `path` to the absolute path of your `drone_dataset/` directory.
- Update `nc` (number of classes) to match the dataset.
- Update `names` to list all class names from the dataset.

### 7. Verify with Your Script

Adapt your `verify_annotation.py` script to load a random image from the downloaded dataset and draw its bounding boxes. Confirm the annotations look correct on the actual drone images.

---

## Checkpoint Questions

1. Why would you apply horizontal flip augmentation but NOT vertical flip for a drone detection dataset? (Hint: think about what a vertically flipped drone image looks like.)

- I am guessing a plane type drone would like almost identical upside down. A propeller drone would like different but it's unlikely to be flying upside down in a real scenario.

2. If your dataset has 1000 drone images and 50 bird images, what problem does this create during training? How would augmentation help?

- I don't remember how it is called exactly but basically it's likely to detect birds as drones due to the lack of data. Augmentation would increase the number of bird images I guess? Though the ratio would stay the same, no?

3. What is the difference between applying augmentations in Roboflow (offline) vs. during training (online)?

- Roboflow augs are limited. can do much more with online training.

---

## Challenge (No Guidance)

**Dataset Statistics Script**

Create `week4_challenge_dataset_stats.py`:

1. Point it at your `drone_dataset/` directory.
2. Scan all label files in `train/`, `val/`, and `test/`.
3. Print a summary report:
   - Total images per split (train/val/test).
   - Total annotations (bounding boxes) per split.
   - Class distribution: how many instances of each class across the entire dataset.
   - Average number of objects per image.
   - Average bounding box size (width × height in normalized coordinates).
4. Flag any potential issues:
   - Images with no annotations (empty `.txt` files).
   - Annotations with coordinates outside the valid range (< 0 or > 1).
   - Extreme aspect ratio boxes (width/height ratio > 10 or < 0.1).

This is a real production task — every ML team runs dataset statistics before training to catch problem early.

---

## Supplemental Reading

**For interviews:**

- **"How do you handle class imbalance?"** — Common interview question. Answers include: oversampling the minority class, undersampling the majority class, augmenting only the minority class, using class-weighted loss functions, or using focal loss (which down-weights easy examples and focuses on hard ones).
- **"What augmentations would you choose for aerial/drone imagery?"** — Rotation, scale variation, brightness/contrast (day/night), and mosaic are the most impactful. Avoid augmentations that create unrealistic images (e.g., extreme color shifts that make the sky purple).

**For production context:**

- **Dataset versioning:** In production, you version your datasets like code. Roboflow does this automatically. If Model v2 performs worse than Model v1, you need to know exactly which dataset each was trained on to debug the regression.
- **Active learning:** A production workflow where the model identifies images it's uncertain about, sends them to human annotators for labeling, and retrains. This creates a feedback loop that continuously improves the dataset with the hardest examples.

**External resources:**

- Roboflow Universe (browse datasets): https://universe.roboflow.com
- Roboflow blog — _"How to Train YOLOv8"_: https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/ — End-to-end tutorial using Roboflow + Ultralytics. A preview of what you'll do in Week 5.
- Albumentations library docs: https://albumentations.ai/docs/ — The go-to Python library for code-based augmentation. You'll use this eventually.

# Challenge Help

You are 100% right, I apologize. Let's back up and break down _how_ to think about extracting this data, step by step.

Right now, your code uses `os.scandir` to count how many _files_ exist. But to know the number of bounding boxes, classes, or errors, we have to actually **open** every single text file and read the text inside it.

### 1. The Core Loop

Instead of a one-liner `sum()`, we need a traditional `for` loop that opens the file:

```python
for entry in os.scandir("month1/week4/drone_dataset/train/labels"):
    if entry.is_file():
        with open(entry.path, "r") as f:
            lines = f.readlines()

        # 'lines' is now a list of strings, where each string is one bounding box!
```

### 2. Counting Boxes & Catching Empty Files

If `len(lines)` is 0, the file is empty. That satisfies your check for "Images with no annotations". You can just append the filename to a list so you know which ones are empty.
Otherwise, the number of boxes in that image is exactly equal to `len(lines)`. You can add that to a running `total_boxes` counter.

### 3. Understanding the Class Distribution

For this, we use a Python **Dictionary** (e.g., `class_counts = {}`).
If we loop through the `lines`, we can split the text just like you did in the previous task: `parts = line.split()`.
`parts[0]` is the class ID (e.g., `"0"` for drone).

We check our dictionary:

- If `"0"` isn't in the dictionary yet, add it: `class_counts["0"] = 1`
- If `"0"` is already there, increment it: `class_counts["0"] += 1`

By the end of scanning all files, your dictionary will look like `{"0": 4500, "1": 120}`, giving you the exact class distribution.

### 4. Mathematical Checks (Aspect Ratio & Out of Bounds)

Still inside that loop over the `parts`, `parts[3]` is width and `parts[4]` is height.

- To find the average box size, just multiply `width * height` and add it to a running `total_area` counter. (At the very end of your script, divide `total_area` by `total_boxes`).
- To check for weird aspect ratios, divide `width / height`. If the result is `> 10` (a very long, thin horizontal box) or `< 0.1` (a very tall, thin vertical box), append the filename to a `weird_boxes` list.
- To check out-of-bounds, just verify if _any_ of the coordinates (x, y, w, h) are `< 0.0` or `> 1.0`.

**Your first step:**
Try writing just the `for` loop that opens the files in the `train/labels` folder, counts the total number of lines (bounding boxes), and prints that number. Don't worry about the math or the dictionaries yet. Just get the file opening and line counting working!
