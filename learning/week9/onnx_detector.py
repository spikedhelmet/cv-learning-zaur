import cv2
import numpy
import onnxruntime as ort
numpy.set_printoptions(suppress=True, precision=3) # to get readable values

class OnnxDetector:
    def __init__(self, model_path, providers, conf_thr=0.25, iou_thr=0.45):
        self.session = ort.InferenceSession(model_path, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

        pass

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        self.height = h
        self.width = w

        scale = min(640/h, 640/w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (640 - new_w) // 2, (640 - new_h) // 2

        resized_frame = cv2.resize(frame, (new_w, new_h))
        canvas = numpy.full((640,640,3), 114, dtype=numpy.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_frame

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        float32 = rgb.astype(numpy.float32) / 255.0
        transposed = float32.transpose(2, 0, 1)
        expanded = numpy.expand_dims(transposed, axis=0)
        tensor = numpy.ascontiguousarray(expanded)

        return tensor, scale, pad_x, pad_y, h, w

    def postprocess(self, output, scale, pad_x, pad_y, orig_h, orig_w):
        # print(output[0].shape)
        tensor = output[0]
        data = output[0][0]
        preds = data.T
        # print(preds[:3])
        boxes = preds[:, :4] 
        scores = preds[:, 4:]
        # print(scores)
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1)
        # print(best_class.shape)
        mask = conf > self.conf_thr
        boxes = boxes[mask]
        conf = conf[mask]
        cls = cls[mask]
        # print("boxes:", boxes)
        if mask.sum() == 0:
            return []
        
        # print("survivors:",mask.sum())

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x = cx - w/2
        y = cy - h/2
        xywh = numpy.stack([x, y, w, h], axis=1)
        # per-class NMS, same as ultralytics (a drone box can't suppress a uav box)
        keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), self.conf_thr, self.iou_thr)
        if len(keep) == 0:
            return []
        keep = numpy.array(keep).flatten()
        # print("keep index:", keep)     # e.g. [0]  or  [0, 7] if there are two objects
        xywh = xywh[keep]
        conf = conf[keep]
        cls = cls[keep]

        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]   # (k,) each

        #remove the shift
        x1 = (x - pad_x) / scale
        y1 = (y - pad_y) / scale
        # undo the shrink
        x2 = (x + w - pad_x) / scale
        y2 = (y + h - pad_y) / scale

        x1 = numpy.clip(x1, 0, orig_w)
        x2 = numpy.clip(x2, 0, orig_w)
        y1 = numpy.clip(y1, 0, orig_h)
        y2 = numpy.clip(y2, 0, orig_h)

        detections = []
        for i in range(len(keep)):
            detections.append((
                float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                float(conf[i]),
                int(cls[i]),
            ))

        return detections

    def detect(self, frame):
        tensor, scale, pad_x, pad_y, h, w = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})

        return self.postprocess(output, scale, pad_x, pad_y, h, w)


# if __name__ == "__main__":
#     detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])

#     cap = cv2.VideoCapture("learning/drone_vid.mp4")

#     while cap.isOpened():
#         ok, frame = cap.read()
#         if not ok:
#             break

#         tensor, scale, pad_x, pad_y, h, w = detector.preprocess(frame)
#         output = detector.session.run(None, {detector.input_name: tensor})
#         detector.postprocess(output, scale=scale,pad_x=pad_x, pad_y=pad_y, orig_h=h, orig_w=w)

#     cap.release()

# if __name__ == "__main__":
#     detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])

#     cap = cv2.VideoCapture("learning/drone_vid.mp4")
#     frames = []

#     while cap.isOpened():
#         ok, frame = cap.read()
#         if not ok:
#             break
        
#         frames.append(frame)
#         if len(frames) >= 50:
#             break
#     cap.release()

#     for f in frames:
#         output = detector.detect(f)
#         print(output)

#         for (x1, y1, x2, y2, conf, cls) in output:
#             cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#         cv2.imshow('frame',f)
#         # print(detector.detect(one_frame))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()




# if __name__ == "__main__":
#     detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])

#     cap = cv2.VideoCapture("learning/drone_vid.mp4")
#     ok, frame = cap.read()
#     cap.release()

#     tensor, scale, pad_x, pad_y, h, w = detector.preprocess(frame)
#     output = detector.session.run(None, {detector.input_name: tensor})
#     detector.postprocess(output, scale=scale,pad_x=pad_x, pad_y=pad_y, orig_h=h, orig_w=w)




