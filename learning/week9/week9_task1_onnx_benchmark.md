# Week 9 — Task 1: Profiling, ONNX Runtime Inference & Benchmarking

---

## Goal & Concept

Your Week 7 pipeline calls `model.track(frame)` and gets everything done by Ultralytics: preprocessing, PyTorch inference, NMS, tracking. It works, but it is a black box and you do not know where the time goes. In this task you will:

1. **Profile** the pipeline stage by stage so optimization targets measured bottlenecks, not guesses.
2. **Export** your drone model to ONNX and run it with **ONNX Runtime**, with no PyTorch and no Ultralytics in the inference path.
3. **Re-implement preprocessing and postprocessing yourself** (letterbox, output decoding, NMS, box un-mapping). This is exactly what you would write in C++ on a production edge device, so doing it once in NumPy gives you the mental model.
4. **Verify correctness**, then **benchmark** PyTorch vs ONNX Runtime with sound methodology.

### Technical explanation

An inference pipeline is a chain of stages, and only one of them is the neural network:

```
decode -> preprocess -> inference -> postprocess (NMS) -> tracking -> draw -> encode
  CPU        CPU          GPU/NPU/CPU        CPU           CPU       CPU     CPU
```

Optimizing the model (quantization, TensorRT) only helps if inference dominates. In your logs, Ultralytics already printed `0.8ms preprocess, 29ms inference, 0.3ms postprocess`, so inference is the biggest slice of the model path, but decode, drawing, JPEG encoding, and database writes sit outside that number. Profile first.

**ONNX** (Open Neural Network Exchange) is a framework-neutral file format: a serialized computation graph plus weights. **ONNX Runtime (ORT)** is an engine that loads that graph and executes it. At session creation ORT applies **graph optimizations** (constant folding, fusing Conv+BatchNorm+activation into one kernel, removing redundant nodes), then dispatches work to an **Execution Provider (EP)**: CPU, CUDA, TensorRT, CoreML, OpenVINO, and so on. The same `.onnx` file can run on a laptop, a Jetson, or an Intel box by changing only the provider list.

**Why Python is a concern (GIL):** CPython's Global Interpreter Lock lets only one thread execute Python bytecode at a time. Two consequences matter here:

- Pure-Python loops are slow (interpreter overhead per operation). Looping in Python over 8400 candidate boxes per frame is far slower than one vectorized NumPy call.
- Threads still help when the heavy work is native code that **releases the GIL** (NumPy, OpenCV, ONNX Runtime, PyTorch all do) or when waiting on I/O. This is why a capture thread plus an inference thread can overlap. Your Week 7 backend already relies on this.

### Defense / C2 context

Counter-drone edge boxes (Jetson Orin, rugged x86 with an NPU) run inference under hard latency budgets, often with no Python at all: C++ with TensorRT or ONNX Runtime. A detection that arrives 300 ms late is a drone that moved several meters. Engineers are expected to know how to measure latency honestly, what each stage costs, and how a model leaves the research framework and reaches the device. That is this week's theme.

### Web / TypeScript analogy

Profiling before optimizing is the same discipline as using the Chrome Performance tab before touching code. ONNX is like compiling TypeScript to a portable target (JS): one artifact, many runtimes. Execution providers resemble polyfills or runtime adapters: the same bundle runs in Node, Deno, or the browser, but the engine underneath differs.

---

## Technical Mechanics & API Overview

### Ultralytics profiling data

- `results[0].speed` returns a dict `{"preprocess": ms, "inference": ms, "postprocess": ms}` for that single call. Values are milliseconds as floats.
- `model.predict(source, device=..., verbose=False, imgsz=640)`: `device` accepts `"cpu"`, `"mps"` (Apple GPU), `"0"` (CUDA GPU 0). `verbose=False` stops the per-frame console print, which would itself distort timing.
- On a Mac, Ultralytics defaults to CPU unless you pass `device="mps"`. Do not assume the GPU is being used.

### Export

`model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=False)`

| Argument        | Meaning                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format="onnx"` | Target format. Returns the path of the written file.                                                                                                              |
| `imgsz=640`     | Input resolution baked into the graph. With `dynamic=False` the input shape is fixed to `(1, 3, 640, 640)`. Fixed shapes let runtimes optimize more aggressively. |
| `opset=17`      | ONNX operator-set version. Higher opsets support newer ops but need a newer runtime. If the runtime is older than the opset, loading fails.                       |
| `simplify=True` | Runs a graph simplifier (onnxslim or onnx-simplifier) to remove redundant ops. Ultralytics may auto-install this dependency on first use.                         |
| `dynamic=False` | Fixed input shape. `True` allows variable batch/size at a performance cost.                                                                                       |
| `half=False`    | Keep FP32 weights. FP16 comes in Task 2.                                                                                                                          |

### ONNX Runtime Python API

- `ort.InferenceSession(path, sess_options=None, providers=[...])`: loads the model. `providers` is an **ordered preference list**; unsupported ops fall back to the next provider (typically CPU).
- `sess.get_providers()`: which providers were actually activated.
- `sess.get_inputs()[0]` and `sess.get_outputs()[0]`: objects with `.name`, `.shape`, `.type`.
- `sess.run(output_names, {input_name: ndarray})`: returns a **list** of NumPy arrays. Pass `None` for `output_names` to get all outputs.
- `ort.SessionOptions()`: `intra_op_num_threads` (threads used inside one op), `graph_optimization_level` (`ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`).
- `ort.get_available_providers()` on this machine returns `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`. CUDA and TensorRT providers are **not** available on Apple Silicon, so TensorRT is conceptual for you this week.

### YOLO11 ONNX output format

At 640x640 the output tensor shape is `(1, 4 + num_classes, 8400)`. Your `drone-model.onnx` has **4 classes** (`0: airplane, 1: bird, 2: drone, 3: uav`, readable from `session.get_modelmeta().custom_metadata_map["names"]`), so its output is **`(1, 8, 8400)`**:

- `1` = batch size (one image per call).
- `8400` candidate predictions = `80*80 + 40*40 + 20*20` (grid cells at strides 8, 16, 32).
- `8` = `4` box values `(cx, cy, w, h)` in **letterboxed input pixel coordinates**, plus `4` class scores, one per class, each already passed through a sigmoid (range 0 to 1).
- There is **no objectness score** in YOLOv8/11 (unlike YOLOv5); the highest class score alone is the confidence.
- The tensor is "channels first": candidates are along the last axis. You must transpose to iterate per candidate.
- **NMS is not included** unless you export with it. Raw output has many overlapping boxes per drone.

### Letterbox

The network expects a square 640x640 input but frames are 16:9. Naive resizing distorts shapes and hurts accuracy. **Letterboxing** resizes preserving aspect ratio, then pads the remaining area with constant gray (114). To map predictions back to the original frame you must undo both the scale and the padding:

```
scale = min(640 / h, 640 / w)
new_w, new_h = round(w * scale), round(h * scale)
pad_x = (640 - new_w) / 2
pad_y = (640 - new_h) / 2
orig_x = (box_x - pad_x) / scale
orig_y = (box_y - pad_y) / scale
```

### Tensor layout

OpenCV frames are `HWC`, `uint8`, `BGR`. The network expects `NCHW`, `float32`, `RGB`, values in `[0, 1]`:
`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` then `astype(np.float32) / 255.0` then `transpose(2, 0, 1)` then `expand_dims(axis=0)`. Use `np.ascontiguousarray` after the transpose, since `transpose` returns a strided view and ORT wants contiguous memory.

### NMS

`cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)`:

- `bboxes`: list of `[x, y, w, h]` (top-left corner plus size, **not** corners)
- `scores`: list of floats, same length
- `score_threshold`: drop candidates below this confidence
- `nms_threshold`: IoU above which a lower-scoring box is suppressed
- Returns the indices of the boxes to keep (a flat array in recent OpenCV versions, nested in some older ones; use `np.array(idx).flatten()` to be safe). When nothing survives it returns an **empty tuple `()`**, not an array.
- `NMSBoxes` is **class-agnostic**: a `bird` box and an overlapping `drone` box suppress each other. `cv2.dnn.NMSBoxesBatched(bboxes, scores, class_ids, score_threshold, nms_threshold)` runs NMS separately per class, which matches Ultralytics' default behavior (`agnostic_nms=False`).

### Benchmarking methodology

- **Warm-up**: the first calls include graph compilation, memory allocation, and (for CoreML) model compilation. Discard the first 10-20 iterations.
- **Clock**: use `time.perf_counter()`, which is monotonic and high-resolution. `time.time()` can jump.
- **Isolate what you measure**: preload frames into a list before timing so video decode does not pollute inference numbers. Time stages separately.
- **Report distribution, not one number**: mean, median (p50), and p95. Latency spikes matter in real-time systems, and the mean hides them.
- **Same inputs for every contender**: identical frames, identical resolution, identical thresholds.
- **Thermal throttling**: Apple Silicon and edge boards throttle under sustained load. Keep runs short and comparable, and note the order you ran things.

---

## Step-by-Step Task

All scripts go in `learning/week9/` and run from the repo root (for example `python learning/week9/profile_pytorch.py`), matching how you run the backend.

### Step 1: Profile the existing PyTorch path (`profile_pytorch.py`)

Load `learning/drone-model.pt` and read the first 300 frames of `learning/drone_vid.mp4` into a Python list (this removes decode time from later measurements). Then, for each frame, call `model.predict(frame, device=..., verbose=False)` and record:

- `results[0].speed["preprocess"]`, `["inference"]`, `["postprocess"]`
- the time of `results[0].plot()` measured yourself with `perf_counter` (drawing cost)
- the time of `cv2.imencode(".jpg", annotated)` (what the MJPEG endpoint pays per frame)

Run it twice, once with `device="cpu"` and once with `device="mps"`. Print a table of mean ms per stage and what percentage of the total each stage is.

**Why:** you cannot optimize what you have not measured. This tells you whether inference, drawing, or encoding is the real cost, and whether the Apple GPU actually helps for a model this small.
**Why discard the first frames:** the first MPS call compiles Metal kernels, so include 10 warm-up frames and exclude them from the stats.
**Why `verbose=False`:** console printing is slow and would be counted in your timing.

### Step 2: Export to ONNX (`export_onnx.py`)

```python
from ultralytics import YOLO
model = YOLO("learning/drone-model.pt")
path = model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
print(path)
```

Move or note the output path (it is written next to the `.pt` file). Open the resulting `.onnx` in **Netron** (https://netron.app, runs in the browser, nothing is uploaded) and look at the graph: find the input node, the output node, and spot the Conv/Sigmoid/Mul blocks.

**Why re-export instead of using `week5/train-2/weights/best.onnx`:** you do not know which arguments produced that file. A benchmark is only fair if the input size, opset, and simplification are known and identical.
**Why `dynamic=False`:** a fixed `(1,3,640,640)` input matches what we preprocess and gives the runtime the most optimization freedom.

### Step 3: Inspect the session

In a scratch script create an `ort.InferenceSession` with `providers=["CPUExecutionProvider"]` and print `get_inputs()[0].name/shape/type` and `get_outputs()[0].name/shape/type`.

**Why:** never write preprocessing from memory. The session tells you the input name you must pass to `run`, the expected dtype (`tensor(float)`), and the output shape. Confirm the output is `[1, 8, 8400]` (your model has 4 classes, so `4 + 4 = 8`) and work out for yourself what each number is before moving on.

### Step 4: Preprocess (`onnx_detector.py`, part 1)

Create a class `OnnxDetector` with `__init__(self, model_path, providers, conf_thr=0.25, iou_thr=0.45)` and a method `preprocess(self, frame)` that returns `(tensor, scale, pad_x, pad_y)`:

1. Compute `scale`, `new_w`, `new_h` as in the Letterbox section.
2. `cv2.resize(frame, (new_w, new_h))`. Note `cv2.resize` takes `(width, height)`, the opposite order of `frame.shape`.
3. Create a `640x640x3` canvas filled with 114 (`np.full((640, 640, 3), 114, dtype=np.uint8)`) and paste the resized image at offset `(pad_y, pad_x)` using integer slicing.
4. BGR to RGB, `float32`, divide by 255, transpose to CHW, add the batch axis, make contiguous.
5. Return the tensor plus `scale` and the pads so postprocessing can undo the mapping.

**Why return scale and pads:** the model reports coordinates in the padded 640x640 space. Without these numbers you cannot draw boxes on the original frame.
**Why the pad offsets must be integers when slicing, but may be fractional in the formula:** a pixel array cannot be indexed with floats, so compute integer pad values once and use the same integers in both places or your boxes will be off by a pixel or two.

### Step 5: Postprocess (`onnx_detector.py`, part 2)

**Input:** whatever `self.session.run(...)` returned, plus the `scale`, `pad_x`, `pad_y` from `preprocess`, plus the original frame size.
**Output:** a plain Python list of detections in **original-frame pixels**:

```
[(x1, y1, x2, y2, conf, cls), ...]     # (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
```

Signature: `postprocess(self, output, scale, pad_x, pad_y, orig_h, orig_w)`. The original size is needed for clipping in 5.7, and `postprocess` has no other way to know it (the model only ever saw the 640x640 canvas).

The model's raw output is 8400 guesses, most of them garbage, in the wrong coordinate system and the wrong box format. Postprocessing is a funnel:

```
8400 candidates
  -> 5.1 reshape so one row = one candidate
  -> 5.2 split box columns from score columns
  -> 5.3 best class + its score per candidate
  -> 5.4 drop low-confidence rows             (~8400 -> a few dozen)
  -> 5.5 center box -> corner box, NMS        (a few dozen -> 1 per object)
  -> 5.6 undo letterbox (padding, then scale)
  -> 5.7 clip to frame
  -> 5.8 Python list of tuples
```

Build it **one sub-step at a time**, and after every line `print(x.shape)` (and for small arrays, the values). Almost every bug in this step is a shape bug, and it is much easier to catch at the line it happens than three lines later.

Open a scratch script, run one real frame through `preprocess` and `session.run`, and keep the result in a variable `output` so you can experiment.

#### 5.1 Unwrap the output and transpose it

`session.run` hands you the result wrapped in **three layers**. Peel them one by one:

| Expression     | Type / shape                      | What this layer is                                                                                                                                                                    |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output`       | Python `list`, `len(output) == 1` | One entry **per output node** of the graph. A model can have several outputs (segmentation models have two). Yours has one, named `output0`. Like a JS function returning `[result]`. |
| `output[0]`    | `ndarray`, `(1, 8, 8400)`         | That output node's tensor. The leading `1` is the **batch** axis: you fed 1 image, so there is 1 result. Mirror image of the `expand_dims(axis=0)` you did in preprocess.             |
| `output[0][0]` | `ndarray`, `(8, 8400)`            | The prediction for image #0 in the batch. This is the actual data.                                                                                                                    |

Now look at what `(8, 8400)` means. It is a table with **8 rows and 8400 columns**. Each **column** is one candidate box; each **row** is one attribute across all candidates:

```
              cand 0   cand 1   cand 2   ...   cand 8399
row 0  cx     312.4    318.9    5.1      ...
row 1  cy     301.0    299.7    4.2      ...
row 2  w       38.2     40.1    9.0      ...
row 3  h       21.5     22.0    8.3      ...
row 4  airplane 0.00     0.01   0.00     ...
row 5  bird    0.02     0.03    0.00     ...
row 6  drone   0.91     0.88    0.00     ...
row 7  uav     0.05     0.04    0.00     ...
```

(Numbers are illustrative.) So to read "everything about candidate 0" you would have to read **down a column**: `data[:, 0]`. That is awkward, because NumPy row-filtering, `cv2.dnn.NMSBoxes`, and the final list all think in terms of **one row per item**.

**Transpose** swaps rows and columns: `(8, 8400)` becomes `(8400, 8)`. Now row `i` is candidate `i`, and the 8 columns are `cx, cy, w, h, airplane, bird, drone, uav`:

```
          cx     cy     w     h    airplane  bird  drone  uav
cand 0   312.4  301.0  38.2  21.5   0.00     0.02  0.91   0.05
cand 1   318.9  299.7  40.1  22.0   0.01     0.03  0.88   0.04
...
cand 8399
```

Toy example to try in a Python shell before touching real data:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3): 2 rows, 3 columns
a.T                          # shape (3, 2): [[1, 4], [2, 5], [3, 6]]
```

Column `[1, 4]` became row `[1, 4]`.

**Trap:** `output[0].T` (without the second `[0]`) gives shape `(8400, 8, 1)`, not `(8400, 8)`. `.T` on an N-dimensional array reverses **all** axes, so the batch axis of size 1 moves to the end instead of disappearing. Drop the batch axis first, then transpose.

**Why the model emits this "sideways" layout at all:** the network computes each attribute as a whole feature map (one Conv output channel = one attribute over every grid cell), so channels-first is its natural layout. Transposing is the postprocessor's job.

**Check:** `preds.shape == (8400, 8)`. Print `preds[:3]` and confirm columns 0-3 look like pixel values (0-640) and columns 4-7 look like probabilities (0-1).

#### 5.2 Split box columns from score columns

Slicing syntax `arr[rows, cols]`, where `:` means "all":

- `preds[:, :4]` = all rows, columns 0 up to (not including) 4 = `cx, cy, w, h`, shape `(8400, 4)`
- `preds[:, 4:]` = all rows, columns 4 to the end = the 4 class scores, shape `(8400, 4)`

**Why `4:` instead of `4:8`:** it keeps the code correct for any class count. Same code works for an 80-class COCO model.

#### 5.3 Pick the best class for each candidate

Each candidate has 4 scores (one per class). You want, per row, **the highest score** (that is the confidence) and **which column it was in** (that is the class id).

- `scores.max(axis=1)`: maximum **across the columns** of each row, giving shape `(8400,)`
- `scores.argmax(axis=1)`: **index** of that maximum, giving shape `(8400,)`, values in `0..3`

`axis=1` means "collapse the column axis". `axis=0` would collapse rows instead and give you one number per class (the best candidate for each class), which is not what you want.

```python
s = np.array([[0.1, 0.7, 0.2],
              [0.9, 0.0, 0.3]])
s.max(axis=1)     # [0.7, 0.9]
s.argmax(axis=1)  # [1,   0  ]   -> row 0 is class 1, row 1 is class 0
```

With your model, `cls == 2` means `drone`. Keep the class id: in Step 6 you will compare it with what PyTorch reports.

#### 5.4 Throw away low-confidence candidates (boolean mask)

`conf > self.conf_thr` does not return one True/False. It compares **every element** and returns a boolean array of the same shape, `(8400,)`. Indexing an array with a boolean array keeps only the positions that are `True`:

```python
conf = np.array([0.02, 0.91, 0.00, 0.88])
mask = conf > 0.25          # [False, True, False, True]
conf[mask]                  # [0.91, 0.88]
boxes[mask]                 # keeps rows 1 and 3 of boxes -> shape (2, 4)
```

Apply the **same** mask to `boxes`, `conf`, and `cls` so that row `i` of each still refers to the same candidate. On a typical frame this shrinks 8400 rows to tens.

**Why filter first:** NMS compares boxes pairwise (cost grows roughly with n²). 8400² is about 70 million comparisons; 30² is 900.
**Why a mask instead of `for i in range(8400): if conf[i] > thr`:** the mask runs as one C loop inside NumPy. The Python loop pays interpreter overhead 8400 times per frame. Write the loop version once, time both with `perf_counter`, and record the ratio (you need it for checkpoint question 4).

**Check:** print `mask.sum()` (number of survivors). On a frame with one drone, expect roughly 5-50. If it is 0 on a frame that clearly has a drone, your preprocessing is wrong (likely BGR/RGB or forgetting `/ 255`).

#### 5.5 Convert to NMS format and run NMS

Why so many survivors for one drone: neighboring grid cells (and neighboring scales) all "see" the same drone and each predicts a box for it. NMS (non-maximum suppression) keeps the highest-scoring box and deletes any other box that overlaps it by more than `iou_thr` IoU (intersection area / union area).

The model gives boxes as **center + size**. `NMSBoxes` wants **top-left corner + size**:

```
          w
   (x,y) +---------+
         |         |
         |  (cx,cy)| h        x = cx - w / 2
         |    *    |          y = cy - h / 2
         +---------+
```

Compute this on the whole column at once (`boxes[:, 0]` is the `cx` column of every surviving row), not row by row. To glue the four resulting columns back into one `(n, 4)` array, use `np.stack([...], axis=1)` or `np.column_stack([...])`.

Then:

```
keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf_thr, self.iou_thr)
```

- `.tolist()` converts NumPy arrays into plain Python lists, which is the most portable thing to pass to OpenCV's Python bindings.
- `keep` holds **indices into the filtered arrays** (from 5.4), not into the original 8400. So `conf[keep[0]]` is the confidence of the first kept box.
- If nothing survived the mask, `keep` is an empty tuple `()`. Return `[]` early if `len(keep) == 0` (or if `mask.sum() == 0` before even calling NMS).
- Wrap with `np.array(keep).flatten()` so you always get a flat int array regardless of OpenCV version.

**Class-agnostic vs per-class:** `NMSBoxes` lets a `drone` box suppress an overlapping `bird` box. Ultralytics does NMS **per class** by default. If in Step 6 you see PyTorch reporting more boxes than you, switch to `cv2.dnn.NMSBoxesBatched(xywh, conf, cls, conf_thr, iou_thr)`, which takes the class ids as an extra argument and only suppresses within the same class.

#### 5.6 Undo the letterbox (map back to the original frame)

Every number so far is in **640x640 canvas pixels**. Preprocess did two things, in this order: (1) shrink by `scale`, (2) shift by `(pad_x, pad_y)`. To undo, reverse the order: **first remove the shift, then undo the shrink.**

```
x1 = (x     - pad_x) / scale
y1 = (y     - pad_y) / scale
x2 = (x + w - pad_x) / scale
y2 = (y + h - pad_y) / scale
```

`(x + w, y + h)` is the bottom-right corner in canvas space. Converting both corners (rather than corner + size) gives you the `(x1, y1, x2, y2)` format directly. Note the width and height are only affected by `scale`, never by padding (padding moves a box; it does not resize it).

**Worked example with your video.** Frames are `1280x720`:

```
scale = min(640/720, 640/1280) = min(0.889, 0.5) = 0.5
new_w, new_h = 640, 360
pad_x = (640 - 640) // 2 = 0
pad_y = (640 - 360) // 2 = 140         # 140 rows of gray above and below the image
```

Suppose NMS kept a box with `x=300, y=290, w=40, h=20` (canvas pixels):

```
x1 = (300      - 0)   / 0.5 = 600
y1 = (290      - 140) / 0.5 = 300
x2 = (300 + 40 - 0)   / 0.5 = 680
y2 = (290 + 20 - 140) / 0.5 = 340
```

In the original frame the box is 80x40 pixels at (600, 300). Sanity check: twice the canvas size, because `scale = 0.5`.

With this video `pad_x` is 0, so a horizontal error would never show up. A bug that mixes up `pad_x` and `pad_y` would only shift boxes vertically by 280 px. Watch for that in your saved images.

#### 5.7 Clip to the frame

A drone at the edge of the frame can get a predicted box that extends a few pixels past the edge, giving `x1 = -3` or `x2 = 1284`. Negative coordinates break array slicing (Python treats `-3` as "3 from the end") and drawing. Clamp: x values to `[0, orig_w]`, y values to `[0, orig_h]`. `np.clip(arr, lo, hi)` does this to a whole array, or use `max(0, min(v, hi))` per value.

#### 5.8 Build the output list

For each index in `keep`, build `(x1, y1, x2, y2, conf, cls)` and append it. A Python loop is fine here: after NMS there are only a handful of boxes.

Convert NumPy scalars to Python types (`float(...)`, `int(...)`). `np.float32` values print as `np.float32(0.91)`, do not serialize to JSON (your Week 7 backend writes detections to a database), and can surprise you in comparisons.

#### 5.9 `detect(self, frame)`

Chain the three stages so callers (Step 6, Step 7, the challenge) only need one call:

```
tensor, scale, pad_x, pad_y = self.preprocess(frame)
output = self.session.run(None, {self.input_name: tensor})
h, w = frame.shape[:2]
return self.postprocess(output, scale, pad_x, pad_y, h, w)
```

#### Done when

1. `detector.detect(frame)` on a frame with a visible drone returns one tuple (or a few, if there are several objects) with `cls == 2` or `3` and a confidence above your threshold.
2. You draw the boxes with `cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)` and save with `cv2.imwrite`, and the box sits **tightly** on the drone.
3. A frame with no objects returns `[]` without crashing.

If the picture looks wrong, use the symptom to locate the bug:

| Symptom                                               | Likely cause                                                                                          |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Box right size, shifted up/down by ~140 px or ~280 px | Padding not subtracted, subtracted twice, or `pad_x`/`pad_y` swapped                                  |
| Box right position-ish, but half or double size       | Multiplied by `scale` instead of dividing (or forgot to divide)                                       |
| Box centered on the bottom-right of the drone         | Forgot the center-to-corner conversion (used `cx, cy` as top-left)                                    |
| `IndexError` / weird shape `(8400, 8, 1)`             | Transposed before removing the batch axis                                                             |
| Zero detections on obvious frames                     | Preprocessing bug: BGR/RGB, missing `/ 255`, or `.T` wrong so you are reading scores from box columns |
| Many overlapping boxes on one drone                   | NMS not applied, or indices from `keep` applied to the unfiltered arrays                              |
| Crash on empty frames                                 | Not handling `NMSBoxes` returning `()`                                                                |

### Step 6: Verify correctness before benchmarking

Run the PyTorch model and your `OnnxDetector` on the same 50 frames and compare:

- number of detections per frame
- for matched boxes, the maximum absolute difference in corner coordinates (expect a few pixels at most) and in confidence (expect about 0.01 or less)

Also draw ONNX boxes on a few frames and save them to disk to check by eye.

**Use identical thresholds on both sides.** Ultralytics' default IoU threshold for `predict` is `0.7`, not the `0.45` your detector uses, so pass them explicitly: `model.predict(frame, conf=0.25, iou=0.45, verbose=False)`. Different thresholds make detection counts differ even when both implementations are correct.

**Why:** a fast but wrong pipeline is worthless, and a bug in letterbox un-mapping gives boxes that look plausible but are systematically shifted. Never publish a benchmark number for an implementation you have not verified.

### Step 7: Benchmark (`benchmark.py`)

Preload 300 frames. Warm up each contender with 20 iterations, then time 300 calls with `perf_counter`. Contenders:

| Contender   | How                                                                       |
| ----------- | ------------------------------------------------------------------------- |
| PyTorch CPU | `model.predict(..., device="cpu")`                                        |
| PyTorch MPS | `model.predict(..., device="mps")`                                        |
| ORT CPU     | `OnnxDetector` with `["CPUExecutionProvider"]`                            |
| ORT CoreML  | `OnnxDetector` with `["CoreMLExecutionProvider", "CPUExecutionProvider"]` |

For each, report mean, p50, p95 latency in ms and FPS (`1000 / mean`). Measure **end to end** (preprocess + inference + postprocess) so the comparison is fair, since Ultralytics' `.speed` numbers include its own pre and post.

For the CoreML run, print `sess.get_providers()` to confirm it was activated. Unsupported ops silently fall back to CPU, so the "CoreML" number might partly be CPU.

**Why p95:** a system averaging 30 ms but spiking to 120 ms every 20th frame will look fine on a mean and feel broken in use.
**Why a separate warm-up for CoreML:** the first `run` compiles the graph for the Apple Neural Engine/GPU and can take seconds.

### Step 8: Session option experiments

With ORT CPU only, vary:

- `graph_optimization_level`: `ORT_DISABLE_ALL` vs `ORT_ENABLE_ALL`
- `intra_op_num_threads`: 1, 2, 4, and the number of performance cores on your machine

Record the effect. **Why:** this shows what graph optimization buys you and that more threads is not always faster (synchronization overhead and efficiency/performance core scheduling on Apple Silicon).

### Step 9: Write up (`learning/week9/benchmark_results.md`)

One table of results plus 3 to 5 sentences answering: Which stage dominates? Did ONNX Runtime beat PyTorch on this hardware, and by how much? Why or why not? What would change on an NVIDIA GPU with TensorRT?

**Why:** being able to explain a measured result is the interview skill. Do not assume ONNX is faster. On x86 CPUs ONNX Runtime commonly beats PyTorch by a clear margin, but on Apple Silicon with MPS the picture is less predictable, and honestly reporting an unexpected result is more valuable than a tidy one.

---

## Checkpoint Questions

1. Your profile shows inference is 30 ms but a full frame in the Week 7 server takes 60+ ms. Name at least four places the remaining time can go.
2. Why does your YOLO11 ONNX output have shape `(1, 8, 8400)`? What does each of the three numbers mean, where does 8400 come from, and why is there no objectness column?
3. Why must the letterbox scale and padding be stored and used in postprocessing? What would the symptom be if you forgot to subtract the padding?
4. Why filter by confidence before NMS, and why is the vectorized mask faster than a Python loop?
5. Why does the GIL not prevent a capture thread and an ONNX Runtime inference thread from overlapping, yet still hurt a Python-loop NMS?
6. What does `get_providers()` returning `['CPUExecutionProvider']` after you requested CoreML tell you?
7. Why discard warm-up iterations and report p95 instead of only the mean?
8. Why is ONNX Runtime not guaranteed to be faster than PyTorch, and what would you check first if it were slower?

---

## Challenge (No Guidance)

**Pipelined, bounded-queue inference.**

Build `pipelined_inference.py` that runs the ONNX detector with a **capture thread** that decodes frames and pushes them into a `queue.Queue(maxsize=2)`, and a main/worker thread that pops frames and runs `detect`.

1. Choose and implement a **drop policy** for when the queue is full (drop the oldest frame, not block). Justify why a real-time system must not let the queue grow unbounded.
2. Measure **end-to-end throughput and per-frame latency** (time from frame read to detections ready) for the pipelined version and for a plain sequential loop. Explain the numbers in terms of what the GIL does and does not block here.
3. Show how latency behaves when you artificially make inference slower (add a `time.sleep`) in the bounded-queue version vs an unbounded one.

---

## Supplemental Reading

### For Interviews

- **GIL:** what it is, why CPython has it (reference-counting memory safety), what releases it (I/O and many native extensions), and when to use threads vs processes. Know that CPython 3.13+ ships an optional free-threaded build (PEP 703), while the default build, including the one in your venv, still has the GIL.
- **Latency vs throughput:** batching raises throughput but adds latency. Real-time tracking is latency-bound, usually batch size 1; offline video analytics is throughput-bound.
- **Why export to ONNX:** decouple training framework from deployment runtime, enable hardware-specific optimizers (TensorRT, OpenVINO, CoreML), remove the heavy PyTorch dependency, enable C++/Rust/mobile deployment.
- **Graph optimizations:** constant folding, operator fusion (Conv + BN + SiLU), layout transformations, dead-node elimination. Be able to explain why fusion helps (fewer kernel launches and fewer memory round-trips).
- **NMS:** algorithm, IoU threshold tradeoff (too low removes real adjacent objects, too high leaves duplicates), and why its cost scales with the number of surviving candidates.
- **Benchmarking pitfalls:** warm-up, caching, thermal throttling, measuring the wrong thing (including decode), averaging away tail latency.

### For Production Context

- **Edge targets:** NVIDIA Jetson (Orin) with TensorRT and DeepStream is the standard for counter-drone edge boxes. Intel hardware uses OpenVINO. Qualcomm and ARM NPUs use vendor runtimes, usually reached through ONNX.
- **C++ inference:** production services call ONNX Runtime or TensorRT from C++ and hand-write letterbox/NMS exactly like you did in NumPy. Python is typically kept for research, tooling, and glue.
- **Operator support and opset compatibility** are the most common deployment failures: a model that exports fine but contains an op the target runtime lacks. Netron and `onnx.checker` are the first debugging tools.
- **Model versioning and reproducible exports:** pin the exact export arguments (`imgsz`, `opset`, `simplify`) in the repo so a deployed artifact can be regenerated.

**Links:**

- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
- ONNX Runtime CoreML provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ONNX Runtime performance tuning: https://onnxruntime.ai/docs/performance/tune-performance/
- Ultralytics export modes and arguments: https://docs.ultralytics.com/modes/export/
- Netron model visualizer: https://netron.app
- Real Python, "What Is the Python Global Interpreter Lock (GIL)?": https://realpython.com/python-gil/
- PEP 703, "Making the Global Interpreter Lock Optional in CPython": https://peps.python.org/pep-0703/
- OpenCV `dnn::NMSBoxes` reference: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
