# A7: Decode the raw YOLO output

**File:** `learning/phaseA/a7_decode.py`  **Time:** 45-60 min  **Needs:** A2, A4, A5, A6

## Concept

This is Week 9 Step 5 again, built from **your own** pieces this time, with no OpenCV NMS.

The model returns a list with one array of shape `(1, 4 + C, 8400)`. For your drone model `C = 4` (`0 airplane, 1 bird, 2 drone, 3 uav`), so it is `(1, 8, 8400)`:

- `1`: batch axis (one image)
- `8`: rows 0-3 are `cx, cy, w, h` in canvas pixels; rows 4-7 are class scores, already sigmoided
- `8400`: candidates, `80*80 + 40*40 + 20*20` grid cells at strides 8, 16, 32

There is no objectness score: the best class score **is** the confidence.

Before writing code, write the pipeline as a list of steps on paper, with the **shape** after each step. Compare it with the Week 9 doc only after you've written your own list.

## Spec

```python
def decode(output, conf_thr=0.25, iou_thr=0.45):
    """output: the (1, 4+C, 8400) array (not the list).
    Returns (boxes, conf, cls): boxes (k, 4) xyxy in CANVAS pixels, conf (k,), cls (k,).
    Uses per-class NMS. Returns three empty arrays if nothing passes conf_thr."""
```

Use your own functions: `best_class`/`filter_by_conf` (A2) or their logic, `cxcywh_to_xyxy` (A4), and `batched_nms` (A5). Import them from your earlier files.

Then write `detect_frame(session, frame)`, which does the whole thing: A6 letterbox, then BGR to RGB, `/255`, CHW, batch axis, contiguous, then `session.run`, `decode`, A6 `boxes_to_original`, and A4 `clip_boxes`.

Expected on **frame 20** of `learning/drone_vid.mp4` (the 21st `cap.read()`):

```
decode(...)        -> boxes [[256.0, 453.6, 314.3, 483.9]]   (canvas)
                      conf  [0.602]
                      cls   [3]
detect_frame(...)  -> boxes [[512.0, 627.3, 628.7, 687.7]]   (original frame)
```

## Check

```python
if __name__ == "__main__":
    import cv2
    import onnxruntime as ort

    session = ort.InferenceSession("learning/drone-model.onnx", providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture("learning/drone_vid.mp4")
    for _ in range(21):
        ok, frame = cap.read()
    cap.release()

    boxes, conf, cls = detect_frame(session, frame)
    assert boxes.shape == (1, 4)
    assert np.allclose(boxes[0], [512.0, 627.3, 628.7, 687.7], atol=0.1)
    assert abs(conf[0] - 0.602) < 0.001 and cls[0] == 3

    empty = np.zeros((1, 8, 8400), dtype=np.float32)       # no candidate passes conf_thr
    b, c, k = decode(empty)
    assert len(b) == 0 and len(c) == 0 and len(k) == 0
    print("A7 ok")
```

Run it from the repo root: `python learning/phaseA/a7_decode.py`.

## Hints

<details><summary>Hint 1 (the first two lines)</summary>

Remove the batch axis **before** transposing. `.T` reverses every axis.

</details>

<details><summary>Hint 2 (empty case)</summary>

If the mask keeps nothing, return early. What should an empty `boxes` array's shape be so that callers can still do `boxes[:, 0]`? `np.zeros((0, 4))` works.

</details>

## Interview questions

1. Where does 8400 come from? What would it be at `imgsz=1280`?
2. YOLOv5 had an objectness column and YOLOv8/11 doesn't. What changed, and how is confidence computed now?
3. Why apply the confidence filter **before** NMS and not after?
4. Your pipeline runs at 30 FPS. Which step would you port to C++ or the GPU first, and how would you find out?

## Rep 2

At least one day later, from a blank file: `decode` only (no letterbox). Target: 15 minutes.
