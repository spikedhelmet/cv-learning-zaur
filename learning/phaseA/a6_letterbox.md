# A6: Letterbox and box mapping

**File:** `learning/phaseA/a6_letterbox.py`  **Time:** 45 min  **Needs:** A2

## Concept

The network takes a square `640x640` image. Frames are rarely square. **Letterboxing** fits the frame inside the square without distorting it:

1. `scale = min(640 / h, 640 / w)`, so the longer side becomes 640.
2. Resize to `new_w = round(w * scale)`, `new_h = round(h * scale)`.
3. Paste it centered on a gray (`114`) canvas. The offsets are `pad_x = (640 - new_w) // 2` and `pad_y = (640 - new_h) // 2`.

The model's boxes come back in **canvas** coordinates. To draw them on the original frame you undo the steps in reverse order: subtract the pad, then divide by the scale.

The best way to test a mapping is a **round trip**: map a box to the canvas and back, and you must get the same box. Here you write both directions, so you can test with a round trip.

## Spec

```python
def letterbox(img, size=640):
    """img: (h, w, 3) uint8 BGR. Returns (canvas, scale, pad_x, pad_y). canvas is (size, size, 3) uint8."""

def boxes_to_canvas(boxes, scale, pad_x, pad_y):
    """xyxy (N, 4) in original-frame pixels -> xyxy in canvas pixels."""

def boxes_to_original(boxes, scale, pad_x, pad_y):
    """xyxy (N, 4) in canvas pixels -> xyxy in original-frame pixels."""
```

| Frame (h x w) | scale | pad_x | pad_y |
| --- | --- | --- | --- |
| 720 x 1280 (your video) | `0.5` | `0` | `140` |
| 1280 x 720 (portrait phone) | `0.5` | `140` | `0` |
| 500 x 333 (odd size) | `1.28` | `107` | `0` |

```
boxes_to_canvas([[600, 300, 680, 340]], 0.5, 0, 140) -> [[300, 290, 340, 310]]
```

## Check

```python
if __name__ == "__main__":
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)                                  # solid red frame
    canvas, scale, px, py = letterbox(img)
    assert canvas.shape == (640, 640, 3) and canvas.dtype == np.uint8
    assert (scale, px, py) == (0.5, 0, 140)
    assert canvas[0, 0].tolist() == [114, 114, 114]       # top padding
    assert canvas[140, 0].tolist() == [0, 0, 255]         # first image row
    assert canvas[499, 0].tolist() == [0, 0, 255]         # last image row
    assert canvas[500, 0].tolist() == [114, 114, 114]     # bottom padding

    assert letterbox(np.zeros((1280, 720, 3), np.uint8))[1:] == (0.5, 140, 0)
    assert letterbox(np.zeros((500, 333, 3), np.uint8))[1:] == (1.28, 107, 0)

    orig = np.array([[600., 300., 680., 340.]])
    on_canvas = boxes_to_canvas(orig, scale, px, py)
    assert np.allclose(on_canvas, [[300, 290, 340, 310]])
    assert np.allclose(boxes_to_original(on_canvas, scale, px, py), orig)   # round trip
    print("A6 ok")
```

Then run it on a real frame: letterbox frame 20 of `learning/drone_vid.mp4`, `cv2.imwrite` the canvas, and look at it. You should see gray bars at the top and bottom.

## Hints

<details><summary>Hint 1</summary>

`cv2.resize(img, (new_w, new_h))` takes **(width, height)**, the opposite order of `img.shape`.

</details>

<details><summary>Hint 2</summary>

Paste with slicing: `canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized`. Rows come first, so y comes first.

</details>

<details><summary>Hint 3 (box mapping)</summary>

Columns 0 and 2 are x and use `pad_x`. Columns 1 and 3 are y and use `pad_y`. `out[:, [0, 2]]` selects both x columns at once.

</details>

## Interview questions

1. Why letterbox instead of a plain resize to 640x640? What happens to a drone's shape with a plain resize?
2. Ultralytics `predict` defaults to `rect=True` (minimal padding, so `384x640` for your video). Why is that faster, and why did it change the confidence in Week 9?
3. You forget to subtract the padding. Which direction are the boxes off, and by how much, on your 1280x720 video?
4. Why must `pad_x` and `pad_y` be integers, and why must preprocess and postprocess use the **same** integers?
5. The model was trained on RGB, and you feed it BGR. Does it crash? What happens instead?

## Rep 2

At least one day later, from a blank file. Target: 15 minutes for all three functions.
