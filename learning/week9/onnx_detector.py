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
        print(mask.sum(), boxes.shape)

        
        # best = preds[:, 4:].max(axis=1).argmax()
        # print(best, preds[best])

        return


if __name__ == "__main__":
    detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])

    cap = cv2.VideoCapture("learning/drone_vid.mp4")
    ok, frame = cap.read()
    cap.release()

    tensor, scale, pad_x, pad_y, h, w = detector.preprocess(frame)
    output = detector.session.run(None, {detector.input_name: tensor})
    detector.postprocess(output, scale=scale,pad_x=pad_x, pad_y=pad_y, orig_h=h, orig_w=w)




