# A1: IoU of two boxes

**File:** `learning/phaseA/a1_iou.py`  **Time:** 20-30 min  **Allowed:** pure Python, paper

## Concept

IoU (Intersection over Union) measures how much two boxes overlap:

```
IoU = area of overlap / area covered by either box
```

It is `1.0` for identical boxes and `0.0` for boxes that don't touch. NMS, mAP, and tracking all use it.

Boxes here are `(x1, y1, x2, y2)`: top-left and bottom-right corners, with y growing **downward** like in images. Area is `(x2 - x1) * (y2 - y1)`.

Two facts to work out yourself on paper:

1. The overlap of two axis-aligned rectangles is **also** an axis-aligned rectangle, or nothing.
2. The union is **not** "area A + area B". Think about what gets counted twice.

## Spec

```python
def iou(a, b):
    """a, b: (x1, y1, x2, y2). Returns a float in [0, 1]."""
```

| a | b | expected |
| --- | --- | --- |
| `(0, 0, 10, 10)` | `(0, 0, 10, 10)` | `1.0` |
| `(0, 0, 10, 10)` | `(5, 5, 15, 15)` | `0.142857...` (25 / 175) |
| `(0, 0, 10, 10)` | `(20, 20, 30, 30)` | `0.0` |
| `(0, 0, 10, 10)` | `(10, 0, 20, 10)` | `0.0` (edges touch, no area) |
| `(0, 0, 10, 10)` | `(2, 2, 4, 4)` | `0.04` (one inside the other) |
| `(0, 0, 30, 10)` | `(10, 0, 40, 10)` | `0.5` |

## Check

Put this at the bottom of your file and run `python learning/phaseA/a1_iou.py`:

```python
if __name__ == "__main__":
    import math
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert math.isclose(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert math.isclose(iou((0, 0, 10, 10), (2, 2, 4, 4)), 0.04)
    assert math.isclose(iou((0, 0, 30, 10), (10, 0, 40, 10)), 0.5)
    # order of arguments must not matter
    assert iou((5, 5, 15, 15), (0, 0, 10, 10)) == iou((0, 0, 10, 10), (5, 5, 15, 15))
    print("A1 ok")
```

## Hints

<details><summary>Hint 1 (direction)</summary>

Compute the intersection rectangle's 4 corners first. Then its area. Then the union.

</details>

<details><summary>Hint 2 (the corners)</summary>

The intersection's left edge is the **rightmost** of the two left edges, and its right edge is the **leftmost** of the two right edges. Same for top and bottom.

</details>

<details><summary>Hint 3 (no overlap)</summary>

If the boxes don't overlap, the intersection's right edge ends up left of its left edge, so the width is negative. A negative width times a negative height gives a positive "area", which is wrong. Clamp width and height at 0 separately.

</details>

## Interview questions

1. Why clamp width and height at 0 **separately**, and not just the final area?
2. Two equal-size boxes are shifted horizontally. What fraction of the width must they overlap to get IoU = 0.5? (Check your answer with the last test case.)
3. Why is IoU used instead of just the intersection area?
4. Some older code computes width as `x2 - x1 + 1`. Why might that be? (Hint: pixel indices vs continuous coordinates.)
5. What should `iou` return if both boxes have zero area? What does your code do?

## Rep 2

Do this at least one day later: delete the file and write it again. No notes, no hints. Target: 10 minutes.
