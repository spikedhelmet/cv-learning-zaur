# A5: NMS from scratch

**File:** `learning/phaseA/a5_nms.py`  **Time:** 60 min  **Needs:** A2, A3, A4

## Concept

A detector predicts many overlapping boxes for one object. NMS keeps the best one and removes the rest:

1. Sort candidates by score, highest first.
2. Take the top one and **keep** it.
3. Remove every remaining candidate whose IoU with it is **greater than** `iou_thr`.
4. Repeat with what's left until nothing is left.

In Week 9 you called `cv2.dnn.NMSBoxes`. Here you write it yourself, using `iou_one_to_many` from A3. "Write NMS" is one of the most common CV coding interview questions.

**Per-class (batched) NMS:** a `drone` box should not remove a `bird` box in the same spot (frame 19 in Week 9). The simple way is to run NMS once per class. There's also a well-known trick torchvision uses: shift each class's boxes by a large offset so boxes of different classes can never overlap, then run normal NMS once. Do **both**.

## Spec

```python
def nms(boxes, scores, iou_thr):
    """boxes (N, 4) xyxy, scores (N,). Returns an int array of kept indices, highest score first."""

def batched_nms(boxes, scores, classes, iou_thr):
    """Same, but boxes only suppress boxes of the same class. classes (N,) ints."""
```

Worked example, so you can trace it by hand first:

```
boxes = [[ 0,  0, 10, 10],   # 0  score 0.90
         [ 1,  1, 11, 11],   # 1  score 0.80  overlaps box 0, IoU 0.68
         [20, 20, 30, 30],   # 2  score 0.70  overlaps box 3, IoU 0.68
         [21, 21, 31, 31],   # 3  score 0.95
         [ 0,  0, 10, 10]]   # 4  score 0.30  identical to box 0

nms(boxes, scores, 0.45)                      -> [3, 0]
batched_nms(..., classes=[0, 1, 0, 0, 0])     -> [3, 0, 1]   (box 1 is a different class, so it survives)
```

Trace it on paper before coding: what gets kept and what gets removed at each round?

## Check

Compares your implementation against OpenCV on 50 random trials of 300 boxes.

```python
if __name__ == "__main__":
    import cv2
    from a4_boxes import xyxy_to_xywh

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [21, 21, 31, 31], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.3])
    assert nms(boxes, scores, 0.45).tolist() == [3, 0]
    assert batched_nms(boxes, scores, np.array([0, 1, 0, 0, 0]), 0.45).tolist() == [3, 0, 1]

    rng = np.random.default_rng(0)
    for trial in range(50):
        xy = rng.uniform(0, 100, (300, 2))
        b = np.concatenate([xy, xy + rng.uniform(1, 50, (300, 2))], axis=1)
        s = rng.uniform(0, 1, 300)
        c = rng.integers(0, 4, 300)

        ref = np.array(cv2.dnn.NMSBoxes(xyxy_to_xywh(b).tolist(), s.tolist(), 0.0, 0.45)).flatten()
        assert sorted(nms(b, s, 0.45).tolist()) == sorted(ref.tolist()), f"nms trial {trial}"

        ref = np.array(cv2.dnn.NMSBoxesBatched(xyxy_to_xywh(b).tolist(), s.tolist(), c.tolist(), 0.0, 0.45)).flatten()
        assert sorted(batched_nms(b, s, c, 0.45).tolist()) == sorted(ref.tolist()), f"batched trial {trial}"
    print("A5 ok")
```

## Hints

<details><summary>Hint 1 (structure)</summary>

Keep an array `order` of indices still in play, sorted by score (A2's `sort_desc`). Loop `while order.size > 0`: the first element is kept, and the rest get filtered.

</details>

<details><summary>Hint 2 (the filter)</summary>

`ious = iou_one_to_many(boxes[order[0]], boxes[order[1:]])`. Then `order[1:][ious <= iou_thr]` is everything that survives this round. That's a boolean mask from A2.

</details>

<details><summary>Hint 3 (the offset trick)</summary>

If every coordinate is below `boxes.max() + 1`, then adding `class_id * (boxes.max() + 1)` to all 4 coordinates of a box moves each class into its own region where it can't touch any other class.

</details>

## Interview questions

1. What's the time complexity of greedy NMS, worst case and typical case? Why does filtering by confidence first help so much?
2. What happens if `iou_thr` is too low? Too high? Give a drone-specific example of each (for example, a swarm flying in formation).
3. Is your loop a Python loop over boxes? How many iterations does it run on frame 20 of the drone video? Why is that acceptable?
4. Explain the offset trick and why it's correct. When could it break? (Think about the offset value.)
5. What is Soft-NMS, in one sentence, and when would you want it?

## Rep 2

At least one day later, from a blank file: `nms` in 10 minutes, `batched_nms` (both versions) in 10 more.
