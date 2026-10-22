# A4: Box formats and clipping

**File:** `learning/phaseA/a4_boxes.py`  **Time:** 30 min  **Needs:** A2

## Concept

The same box can be written three common ways, and every library picks a different one:

| Name | Values | Used by |
| --- | --- | --- |
| `xyxy` | top-left x, y, bottom-right x, y | IoU, drawing, Ultralytics `boxes.xyxy`, torchvision NMS |
| `cxcywh` | center x, y, width, height | YOLO raw output, YOLO label `.txt` files (normalized) |
| `xywh` | top-left x, y, width, height | OpenCV (`cv2.dnn.NMSBoxes`, `cv2.Rect`), COCO JSON |

A large share of real detection bugs are format mix-ups: a box drawn shifted by half its size, or an IoU computed on the wrong format. The fix is to write the conversion functions once, test them, and use only them.

**Clipping:** after mapping to the original frame, a box can stick out past the edge. x values clip to `[0, w]` and y values to `[0, h]`, two different ranges.

## Spec

All functions take `(N, 4)` and return a **new** `(N, 4)` array. No loops, and don't modify the input.

```python
def cxcywh_to_xyxy(b): ...
def xyxy_to_cxcywh(b): ...
def xyxy_to_xywh(b): ...
def xywh_to_xyxy(b): ...
def clip_boxes(b, h, w): ...   # xyxy in, xyxy out
```

```
xyxy = [[10, 20, 50, 40],
        [ 0,  0,  4,  4]]

xyxy_to_cxcywh(xyxy) -> [[30, 30, 40, 20],
                         [ 2,  2,  4,  4]]
xyxy_to_xywh(xyxy)   -> [[10, 20, 40, 20],
                         [ 0,  0,  4,  4]]

clip_boxes([[-5, 10, 1300, 800]], h=720, w=1280) -> [[0, 10, 1280, 720]]
```

## Check

```python
if __name__ == "__main__":
    xyxy = np.array([[10., 20., 50., 40.], [0., 0., 4., 4.]])
    assert np.allclose(xyxy_to_cxcywh(xyxy), [[30, 30, 40, 20], [2, 2, 4, 4]])
    assert np.allclose(xyxy_to_xywh(xyxy), [[10, 20, 40, 20], [0, 0, 4, 4]])

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (100, 2))
    r = np.concatenate([xy, xy + rng.uniform(1, 50, (100, 2))], axis=1)
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(r)), r)   # round trips
    assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(r)), r)

    before = r.copy()
    clip_boxes(r, 50, 50)
    assert np.array_equal(r, before), "clip_boxes modified its input"
    assert np.allclose(clip_boxes(np.array([[-5., 10., 1300., 800.]]), h=720, w=1280), [[0, 10, 1280, 720]])
    print("A4 ok")
```

## Hints

<details><summary>Hint 1</summary>

Pull out the columns (`x1 = b[:, 0]` etc.), compute the new columns, then `np.stack([...], axis=1)`. You did exactly this in `onnx_detector.py`.

</details>

<details><summary>Hint 2 (clip without modifying the input)</summary>

`out = b.copy()` first. `out[:, [0, 2]]` selects the x1 and x2 columns together.

</details>

## Interview questions

1. YOLO label files store `cx cy w h` normalized to `[0, 1]`. Write the conversion to pixel `xyxy` for a 1280x720 image, out loud.
2. Why does the "modifies its input" check matter? Describe a bug it would cause in a pipeline.
3. Why convert to `xywh` before calling `cv2.dnn.NMSBoxes`? What happens if you pass `xyxy` instead? (Think about what the "width" becomes.)

## Rep 2

At least one day later, from a blank file. Target: 10 minutes.
