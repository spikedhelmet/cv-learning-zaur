# Week 5 — Task 1: Training Your First Custom YOLO Model

---

## Goal

Train a YOLO model on **your drone detection dataset** from Week 4. By the end of this task, you will have a model that was trained specifically to detect drones — not the generic 80-class COCO model you've been using, but *your* model that you trained yourself. You'll understand transfer learning, training hyperparameters, and how to read training metrics.

---

## Concept: Transfer Learning vs. Training From Scratch

There are two ways to train a neural network:

### Training From Scratch
You initialize the model with **random weights** (random numbers). The model knows absolutely nothing. It must learn everything — what an edge is, what a shape is, what a texture is, what a drone looks like — all from your dataset alone.

**Problem:** This requires millions of images and days/weeks of GPU time. You don't have either.

### Transfer Learning (Fine-Tuning)
You start with a model that was **already trained on a huge dataset** (like COCO, with 330,000 images across 80 classes). This model already knows what edges, shapes, textures, and common objects look like. Its "brain" (weights) already encodes general visual knowledge.

You then **fine-tune** it on your specific dataset. You're essentially saying: *"You already know how to see. Now learn what a drone looks like specifically."*

**This is what you'll do.** You'll start with `yolo11n.pt` (the pre-trained Nano model) and fine-tune it on your drone dataset. The model keeps all its general visual knowledge and just learns to focus on drones.

### Analogy
- **From scratch:** Teaching a baby to recognize a drone. First they need to learn what "seeing" is, then shapes, then objects, then finally drones. Takes years.
- **Transfer learning:** Teaching an experienced photographer to spot drones. They already understand images, lighting, depth, objects. They just need a few examples of drones and they're good. Takes hours.

### Defense Context
In defense, you almost never train from scratch. You take a model pre-trained on a large public dataset, then fine-tune it on your classified/proprietary data (e.g., specific drone models, thermal imagery, radar signatures). This is faster, cheaper, and works with smaller datasets.

---

## Technical Mechanics

### The Training Command

With Ultralytics, training is a single Python call:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load pre-trained weights (transfer learning)

results = model.train(
    data="path/to/data.yaml",  # Points to your dataset
    epochs=50,                 # How many times to loop through all training images
    imgsz=640,                 # Resize all images to 640x640 before training
    batch=16,                  # Process 16 images at a time
)
```

That's it. Ultralytics handles everything: data loading, augmentation, optimization, validation after each epoch, saving checkpoints, and generating metrics.

### Key Hyperparameters

| Parameter | What it controls | Default | Guidance |
|---|---|---|---|
| `epochs` | How many full passes through the training set | 100 | Start with 50. If the model is still improving at epoch 50, increase. |
| `imgsz` | Input image resolution (resized before training) | 640 | 640 is standard. Larger = better accuracy but slower and more memory. |
| `batch` | How many images to process simultaneously | 16 | Depends on your GPU/RAM. If you get "out of memory" errors, reduce to 8 or 4. |
| `lr0` | Initial learning rate | 0.01 | Controls how aggressively the model updates its weights. Too high = unstable, too low = slow learning. Leave default for now. |
| `patience` | Early stopping: stop training if no improvement for N epochs | 50 | Prevents wasting time if the model has converged. |

### What Happens During Training

For each epoch, the training loop does this:
1. **Shuffle** the training images randomly.
2. **Load a batch** of images (e.g., 16 at a time).
3. **Apply online augmentations** (mosaic, flip, color jitter — built into YOLO).
4. **Forward pass:** Feed the batch through the model, get predictions.
5. **Compute loss:** Compare predictions to the ground truth labels. The loss is a number that measures "how wrong" the model is.
6. **Backward pass (backpropagation):** Calculate how to adjust each weight to reduce the loss.
7. **Update weights:** Nudge the weights slightly in the right direction.
8. **Repeat** for all batches in the training set. That's 1 epoch.
9. **Validate:** After each epoch, run the model on the `val` set (no weight updates) and compute metrics.

### Training Output: What to Watch

During training, YOLO prints a table after each epoch. The key columns:

| Metric | What it means |
|---|---|
| `box_loss` | How wrong the predicted bounding box positions are. Should decrease over time. |
| `cls_loss` | How wrong the class predictions are. Should decrease over time. |
| `mAP50` | Mean Average Precision at IoU threshold 0.50. This is your primary accuracy metric. Higher = better. Range: 0.0 to 1.0. |
| `mAP50-95` | mAP averaged across IoU thresholds 0.50 to 0.95. Stricter than mAP50. |
| `precision` | Of all the boxes the model predicted as "drone", what fraction were actually drones? |
| `recall` | Of all the actual drones in the images, what fraction did the model find? |

**IoU (Intersection over Union):** Measures how well a predicted box overlaps with the ground truth box. IoU = 1.0 means perfect overlap. IoU = 0.5 means 50% overlap, which is the standard "good enough" threshold. mAP50 uses this 50% threshold.

### Overfitting: How to Spot It

**Overfitting** = the model memorizes the training data instead of learning general patterns.

Signs:
- **Training loss keeps decreasing** but **validation mAP stops improving** (or gets worse).
- The model detects drones perfectly on training images but misses them on new images it hasn't seen.

This is why the validation set exists: it's your early warning system. If val metrics plateau while training metrics keep improving, you're overfitting.

**Fixes:**
- Stop training earlier (reduce epochs or use `patience`).
- Add more data or more augmentation.
- Use a smaller model (Nano instead of Small).

### Where Results Are Saved

After training, YOLO saves everything to `runs/detect/train/`:
- `weights/best.pt` — The model weights from the epoch with the highest mAP. **This is your trained model.**
- `weights/last.pt` — The model weights from the final epoch.
- `results.csv` — All metrics from every epoch (you can plot these).
- `confusion_matrix.png` — Shows what the model gets right and wrong.
- `results.png` — Plots of loss and mAP over time.
- `val_batch0_pred.png` — Sample predictions on validation images.

---

## Step-by-Step Task: Train Your Drone Detector

### 1. Verify Your Dataset

Before training, make sure your `data.yaml` has the correct absolute path and that your images/labels folders are populated. Run your `dataset_stats.py` one more time to confirm everything looks right.

### 2. Check GPU Availability

Run this in Python to see if you have a GPU available:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU — will use CPU")
```

**If you have a GPU:** Training will take 20-60 minutes depending on the GPU.
**If you don't have a GPU (CPU only):** Training will be very slow (hours). You have two options:
- Train with fewer epochs (e.g., 10-20) and a smaller image size (e.g., `imgsz=320`) just to see the process work.
- Use **Google Colab** (free GPU). I can help you set that up if needed.

### 3. Create `week5_train.py`

Create `month1/week5/week5_train.py`:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="C:/Users/user/Desktop/Computer Vision/month1/week4/drone_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    verbose=True,
)
```

**Important:** The `data` path must be the **absolute path** to your `data.yaml`. Relative paths often break during training because YOLO changes the working directory internally.

### 4. Run Training

```bash
python month1/week5/week5_train.py
```

Watch the output. After each epoch you'll see the loss values and mAP. Training is working correctly if:
- `box_loss` and `cls_loss` decrease over the first 10-20 epochs.
- `mAP50` increases over the first 10-20 epochs.

### 5. Examine the Results

After training finishes, go to the `runs/detect/train/` folder and look at:
- `results.png` — Are the loss curves going down? Is mAP going up?
- `confusion_matrix.png` — What does the model confuse?
- `val_batch0_pred.png` — Do the predicted boxes look reasonable on real images?

### 6. Test Your Trained Model

Create `month1/week5/week5_test_trained.py`:
- Load your trained model: `model = YOLO("runs/detect/train/weights/best.pt")`
- Run it on your webcam (like you did in Week 3) and see if it detects drones in real-time.
- If you don't have a real drone, find drone images/videos online and test on those.

---

## Checkpoint Questions

1. What is the difference between `yolo11n.pt` (pre-trained) and training from scratch? What would happen if you used `YOLO("yolo11n.yaml")` instead of `YOLO("yolo11n.pt")`?
2. If your `box_loss` is decreasing but your `mAP50` is not increasing, what might be going wrong?
3. What does `patience=10` do? Why is it useful?
4. After training, why do we use `best.pt` instead of `last.pt` for deployment?

---

## Challenge (No Guidance)

**Training Comparison**

1. Train the same dataset twice:
   - Once with `yolo11n.pt` (Nano)
   - Once with `yolo11s.pt` (Small)
2. Compare the final `mAP50`, training time, and model file size (`best.pt`).
3. Write your findings in a text file: `week5_comparison.txt`. Which model would you choose for a drone-mounted camera? Which for a ground station with a powerful GPU?

---

## Supplemental Reading

**For interviews:**
- **"Explain transfer learning."** — Start with: "Instead of training from random weights, we initialize the model with weights pre-trained on a large dataset like COCO. The early layers already encode general visual features (edges, textures, shapes). We fine-tune the later layers on our specific task. This requires less data and converges faster."
- **"What is mAP and how is it calculated?"** — mAP (mean Average Precision) averages the precision-recall curve across all classes. At IoU=0.50, a predicted box is "correct" if it overlaps with the ground truth by at least 50%. mAP50-95 averages across multiple IoU thresholds (0.50, 0.55, ..., 0.95), making it a stricter metric.
- **"How do you prevent overfitting?"** — Early stopping (patience), data augmentation, dropout, weight decay, using a smaller model, or collecting more data.

**For production context:**
- **ONNX export:** After training, you can export your model to ONNX format (`model.export(format="onnx")`), which is a universal format that runs on any hardware (NVIDIA, Intel, ARM, browsers). This is how models move from research to production.
- **TensorRT:** For NVIDIA GPUs, converting ONNX to TensorRT can give 2-5x inference speedup. This is standard practice for real-time systems.
- **Model versioning:** In production, every trained model is versioned with its dataset version, hyperparameters, and metrics. If Model v3 performs worse than Model v2, you need to trace back exactly what changed.

**External resources:**
- Ultralytics Training Docs: https://docs.ultralytics.com/modes/train/ — Official reference for all training parameters.
- *"Understanding mAP"* — Jonathan Hui (Medium): A clear visual explanation of precision, recall, AP, and mAP with worked examples.
- Google Colab: https://colab.research.google.com — Free GPU access for training if your local machine is too slow.
