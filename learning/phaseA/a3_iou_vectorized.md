# A3: Vectorized IoU

**File:** `learning/phaseA/a3_iou_vectorized.py`  **Time:** 45-60 min  **Needs:** A1, A2

## Concept

NMS compares one box against all remaining boxes. Tracking compares every track against every detection. Doing that with A1's `iou` in a Python loop is slow, so here you compute it with array operations.

It's the same formula as A1. Each scalar operation becomes an array operation:

| A1 (scalars) | A3 (arrays) |
| --- | --- |
| `max(a, b)` | `np.maximum(a, b)` (elementwise, broadcasts) |
| `min(a, b)` | `np.minimum(a, b)` |
| `max(0, w)` | `np.clip(w, 0, None)` or `np.maximum(w, 0)` |
| `box[0]` | `boxes[:, 0]` (column of all x1s) |

**One-to-many:** one box `(4,)` against `N` boxes `(N, 4)` gives `(N,)`. A scalar broadcasts against a column without any extra work.

**Many-to-many:** `(N, 4)` against `(M, 4)` gives an `(N, M)` matrix. This is A2's `pairwise_diff` idea: make `a`'s columns `(N, 1)` and `b`'s columns `(1, M)`.

## Spec

```python
def iou_one_to_many(box, boxes):
    """box (4,), boxes (N, 4), both xyxy. Returns (N,)."""

def iou_matrix(a, b):
    """a (N, 4), b (M, 4), both xyxy. Returns (N, M) with out[i, j] = iou(a[i], b[j])."""
```

No Python loops in either function.

```
iou_one_to_many([0, 0, 10, 10], [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
-> [1.0, 0.142857, 0.0]
```

## Check

This compares your vectorized version against your own A1 on random boxes. If A1 is right and the two disagree, the bug is in A3.

```python
if __name__ == "__main__":
    from a1_iou import iou

    rng = np.random.default_rng(0)

    def random_boxes(n):
        xy = rng.uniform(0, 100, size=(n, 2))
        wh = rng.uniform(1, 50, size=(n, 2))
        return np.concatenate([xy, xy + wh], axis=1)

    small = iou_one_to_many(np.array([0, 0, 10, 10]),
                            np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]]))
    assert np.allclose(small, [1.0, 25 / 175, 0.0])

    box, boxes = random_boxes(1)[0], random_boxes(200)
    out = iou_one_to_many(box, boxes)
    assert out.shape == (200,)
    assert np.allclose(out, [iou(box, b) for b in boxes])

    a, b = random_boxes(30), random_boxes(40)
    m = iou_matrix(a, b)
    assert m.shape == (30, 40)
    assert np.allclose(m, [[iou(x, y) for y in b] for x in a])
    print("A3 ok")
```

## Hints

<details><summary>Hint 1</summary>

Start from your A1 code. Replace each `max`/`min` with `np.maximum`/`np.minimum`, and each `b[0]` with `boxes[:, 0]`. For one-to-many, that's almost all you need.

</details>

<details><summary>Hint 2 (matrix)</summary>

`a[:, 0]` has shape `(N,)`. `a[:, None, 0]` has shape `(N, 1)`. `b[None, :, 0]` has shape `(1, M)`. `np.maximum` of those two is `(N, M)`.

</details>

<details><summary>Hint 3 (areas)</summary>

The area of each box in `a` is `(N,)` and in `b` is `(M,)`. For the union you need `(N, M)`: the same broadcast trick again.

</details>

## Interview questions

1. What is the memory cost of `iou_matrix` for 8400 x 8400 boxes in float64? Why does that matter for NMS?
2. Where in tracking do you need an `(N, M)` IoU matrix, and what do you do with it next?
3. Your one-to-many version also works if `box` has shape `(1, 4)`? Try it. Why or why not?

## Rep 2

At least one day later, from a blank file, no notes. Target: 15 minutes for both functions.
