# A2: NumPy fluency

**File:** `learning/phaseA/a2_numpy.py`  **Time:** 60-90 min  **Allowed:** NumPy docs for function signatures

## Concept

This drill covers almost everything that made Week 9 Step 5 hard. There are five ideas:

1. **Shape** is the array's size along each axis. `(8400, 8)` means 8400 rows and 8 columns.
2. **axis=k** in a reduction (`sum`, `max`, `argmax`) **removes** axis `k`. `(8400, 4).max(axis=1)` gives `(8400,)`, one value per row.
3. **Boolean mask:** `arr[mask]` keeps the positions where `mask` is `True`. For a 2D array, a 1D mask keeps whole rows.
4. **Integer (fancy) indexing:** `arr[[3, 0, 7]]` picks those rows in that order. This is how you apply `keep` after NMS.
5. **Broadcasting:** when shapes differ, axes of size 1 are stretched to match. `(N, 1) - (1, M)` gives `(N, M)`, every pair. This is how you get IoU for every pair of boxes without a loop.

## Part 1: predict, then run

For each line, **write your predicted output as a comment first**, then run it and compare. Getting a prediction wrong is the useful part, so don't run first.

```python
import numpy as np
a = np.arange(24).reshape(2, 3, 4)

a.shape
a[0].shape
a[:, 0].shape
a[..., 0].shape
a.T.shape
a.transpose(0, 2, 1).shape
a.sum(axis=0).shape
a.max(axis=-1).shape
a.reshape(-1, 4).shape
a[None].shape
a[:, :, :2].shape
a[0, 1]
a.max(axis=-1)
```

<details><summary>Answers (only after you predicted all of them)</summary>

```
(2, 3, 4)
(3, 4)
(2, 4)
(2, 3)
(4, 3, 2)
(2, 4, 3)
(3, 4)
(2, 3)
(6, 4)
(1, 2, 3, 4)
(2, 3, 2)
[4 5 6 7]
[[ 3  7 11]
 [15 19 23]]
```

`a.T` reverses **all** axes. That is the Week 9 trap where `output[0].T` gave `(8400, 8, 1)`.

</details>

## Part 2: write these functions

No Python `for` loops in any of them.

```python
def best_class(scores):
    """scores: (N, C). Returns (conf, cls), both (N,): max score per row and its column index."""

def filter_by_conf(boxes, conf, cls, thr):
    """boxes (N, 4), conf (N,), cls (N,). Keep rows where conf > thr. Returns all three, filtered."""

def sort_desc(values):
    """values (N,). Returns the indices that sort values from highest to lowest."""

def pairwise_diff(a, b):
    """a (N,), b (M,). Returns (N, M) where out[i, j] = a[i] - b[j]."""
```

Expected:

```
scores = [[0.1, 0.7, 0.2],
          [0.9, 0.0, 0.3],
          [0.2, 0.2, 0.6],
          [0.05, 0.1, 0.15]]
best_class(scores)             -> conf [0.7, 0.9, 0.6, 0.15], cls [1, 0, 2, 2]

sort_desc([3, 1, 2])           -> [0, 2, 1]

pairwise_diff([1, 5, 3], [10, 20]) -> [[ -9, -19],
                                       [ -5, -15],
                                       [ -7, -17]]
```

## Part 3: `np.max` vs `np.maximum`

These two are easy to confuse, and A3 needs the right one. Predict, then run:

```python
np.max(np.array([1, 5, 3]))           # ?
np.maximum(np.array([1, 5, 3]), 2)    # ?
np.maximum(np.array([1, 5, 3]), np.array([4, 4, 4]))  # ?
```

In one sentence each, write what each function does.

## Check

```python
if __name__ == "__main__":
    scores = np.array([[0.1, 0.7, 0.2], [0.9, 0.0, 0.3], [0.2, 0.2, 0.6], [0.05, 0.1, 0.15]])
    conf, cls = best_class(scores)
    assert np.allclose(conf, [0.7, 0.9, 0.6, 0.15]) and cls.tolist() == [1, 0, 2, 2]

    boxes = np.arange(16).reshape(4, 4)
    b, c, k = filter_by_conf(boxes, conf, cls, 0.5)
    assert b.shape == (3, 4) and b[:, 0].tolist() == [0, 4, 8]
    assert np.allclose(c, [0.7, 0.9, 0.6]) and k.tolist() == [1, 0, 2]

    assert sort_desc(np.array([3, 1, 2])).tolist() == [0, 2, 1]

    d = pairwise_diff(np.array([1, 5, 3]), np.array([10, 20]))
    assert d.shape == (3, 2) and d.tolist() == [[-9, -19], [-5, -15], [-7, -17]]
    print("A2 ok")
```

## Hints

<details><summary>Hint: sort_desc</summary>

`np.argsort` only sorts ascending. Is there something you can do to the values so that ascending order on them is descending order on the originals?

</details>

<details><summary>Hint: pairwise_diff</summary>

Turn `a` into a column `(N, 1)` and `b` into a row `(1, M)`. `a[:, None]` adds an axis. Then just subtract.

</details>

## Interview questions

1. `boxes` is `(8400, 4)` and `mask` is `(8400,)`. What is the shape of `boxes[mask]`, and why does it keep whole rows?
2. Why is a vectorized mask faster than `for i in range(8400): if conf[i] > thr`? Name the actual reason, not "NumPy is fast".
3. What does broadcasting `(3, 1)` with `(4,)` give, and why?
4. `a.T` vs `a.transpose(1, 0)` vs `a.swapaxes(0, 1)`: when are they the same, and when does `.T` surprise you?

## Rep 2

At least one day later: Part 2 again from a blank file, no notes. Target: 15 minutes.
