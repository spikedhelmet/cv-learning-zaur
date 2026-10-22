# A8: Phase A exit test

**File:** `learning/phaseA/a8_detector.py`  **Time limit:** 45 minutes  **Do this only after A1-A7 have passed Rep 2.**

## Rules

- Blank file. Don't open A1-A7, `onnx_detector.py`, or the Week 9 doc.
- Allowed docs: API **signatures** only (`ort.InferenceSession`, `cv2.resize`, `np.argsort`, ...).
- No OpenCV NMS. Use your own.
- Everything in **one** file (no imports from your drill files). In an interview you won't have them.
- Start a timer.

## Task

Write a class:

```python
class Detector:
    def __init__(self, model_path, providers, conf_thr=0.25, iou_thr=0.45): ...
    def detect(self, frame):
        """BGR frame of any size -> list of (x1, y1, x2, y2, conf, cls) in original-frame pixels."""
```

It must contain, written from memory: IoU (vectorized), per-class NMS, letterbox, box mapping back to the frame, clipping, and decoding.

## Pass criteria

1. **Correctness:** copy `learning/week9/verify.py` to `learning/phaseA/a8_verify.py`, change it to use your `Detector`, and run it. Required result: `count mismatches: 0`, `class mismatches: 0`, max corner diff < 0.5 px, max confidence diff < 0.001.
2. **Empty frame:** `detect(np.zeros((720, 1280, 3), np.uint8))` returns `[]` without crashing.
3. **Time:** both of the above within 45 minutes. Record your time here: ______

## Then: mock interview

Ask Claude to run a 15-minute mock interview on Phase A. It will pick questions from the interview sections of A1-A7 and add follow-ups. You pass when you can answer without looking anything up.

## If you fail

That's useful information, not a setback. Note which part blocked you, redo that drill's Rep 2, and retake A8 at least one day later.
