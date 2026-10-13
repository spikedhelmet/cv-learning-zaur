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

| Argument | Meaning |
|---|---|
| `format="onnx"` | Target format. Returns the path of the written file. |
| `imgsz=640` | Input resolution baked into the graph. With `dynamic=False` the input shape is fixed to `(1, 3, 640, 640)`. Fixed shapes let runtimes optimize more aggressively. |
| `opset=17` | ONNX operator-set version. Higher opsets support newer ops but need a newer runtime. If the runtime is older than the opset, loading fails. |
| `simplify=True` | Runs a graph simplifier (onnxslim or onnx-simplifier) to remove redundant ops. Ultralytics may auto-install this dependency on first use. |
| `dynamic=False` | Fixed input shape. `True` allows variable batch/size at a performance cost. |
| `half=False` | Keep FP32 weights. FP16 comes in Task 2. |

### ONNX Runtime Python API

- `ort.InferenceSession(path, sess_options=None, providers=[...])`: loads the model. `providers` is an **ordered preference list**; unsupported ops fall back to the next provider (typically CPU).
- `sess.get_providers()`: which providers were actually activated.
- `sess.get_inputs()[0]` and `sess.get_outputs()[0]`: objects with `.name`, `.shape`, `.type`.
- `sess.run(output_names, {input_name: ndarray})`: returns a **list** of NumPy arrays. Pass `None` for `output_names` to get all outputs.
- `ort.SessionOptions()`: `intra_op_num_threads` (threads used inside one op), `graph_optimization_level` (`ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`).
- `ort.get_available_providers()` on this machine returns `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`. CUDA and TensorRT providers are **not** available on Apple Silicon, so TensorRT is conceptual for you this week.

### YOLO11 ONNX output format

For a single-class model at 640x640, the output tensor shape is `(1, 5, 8400)`:
- `8400` candidate predictions = `80*80 + 40*40 + 20*20` (grid cells at strides 8, 16, 32).
- `5` = `4` box values `(cx, cy, w, h)` in **letterboxed input pixel coordinates**, plus `1` class score (`4 + num_classes` in general).
- There is **no objectness score** in YOLOv8/11 (unlike YOLOv5); the class score alone is the confidence.
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
- Returns the indices of the boxes to keep (a flat array in recent OpenCV versions, nested in some older ones; use `np.array(idx).flatten()` to be safe).

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

**Why:** never write preprocessing from memory. The session tells you the input name you must pass to `run`, the expected dtype (`tensor(float)`), and the output shape. Confirm the output is `[1, 5, 8400]` (or `[1, 4+nc, 8400]`) and work out for yourself what each number is before moving on.

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

Write `postprocess(self, output, scale, pad_x, pad_y)` that takes the raw `(1, 4+nc, 8400)` array and returns a list of `(x1, y1, x2, y2, conf, cls)` in original-frame pixels.

Pseudocode:

```
preds = output[0].T                      # (8400, 4+nc): one row per candidate
boxes = preds[:, :4]                     # cx, cy, w, h
class_scores = preds[:, 4:]
conf = class_scores.max(axis=1)
cls = class_scores.argmax(axis=1)
mask = conf > conf_thr                   # boolean filter FIRST (cheap, vectorized)
boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
convert (cx, cy, w, h) -> (x, y, w, h) with x = cx - w/2, y = cy - h/2
idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), conf.tolist(), conf_thr, iou_thr)
for each kept index: undo pad and scale, clip to frame bounds
```

**Why filter by confidence before NMS:** NMS is expensive (pairwise IoU). Dropping the ~8390 near-zero candidates with one vectorized mask leaves a handful to suppress.
**Why this is vectorized and not a Python `for` loop over 8400 rows:** this is the GIL/interpreter-overhead lesson in practice. Try the naive loop once and time it to see the difference.
**Why clip:** after un-mapping, boxes at the image border can extend a few pixels outside the frame.

Add a `detect(self, frame)` method that chains preprocess, `self.session.run`, and postprocess.

### Step 6: Verify correctness before benchmarking

Run the PyTorch model and your `OnnxDetector` on the same 50 frames and compare:
- number of detections per frame
- for matched boxes, the maximum absolute difference in corner coordinates (expect a few pixels at most) and in confidence (expect about 0.01 or less)

Also draw ONNX boxes on a few frames and save them to disk to check by eye.

**Why:** a fast but wrong pipeline is worthless, and a bug in letterbox un-mapping gives boxes that look plausible but are systematically shifted. Never publish a benchmark number for an implementation you have not verified.

### Step 7: Benchmark (`benchmark.py`)

Preload 300 frames. Warm up each contender with 20 iterations, then time 300 calls with `perf_counter`. Contenders:

| Contender | How |
|---|---|
| PyTorch CPU | `model.predict(..., device="cpu")` |
| PyTorch MPS | `model.predict(..., device="mps")` |
| ORT CPU | `OnnxDetector` with `["CPUExecutionProvider"]` |
| ORT CoreML | `OnnxDetector` with `["CoreMLExecutionProvider", "CPUExecutionProvider"]` |

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
2. Why does the YOLO11 ONNX output have shape `(1, 5, 8400)` and where does 8400 come from? Why is there no objectness column?
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
