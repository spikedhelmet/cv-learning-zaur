import cv2
import numpy
import onnxruntime as ort

class OnnxDetector:
    def __init__(self, model_path, providers, conf_thr=0.25, iou_thr=0.45):
        self.session = ort.InferenceSession("learning/drone-model.onnx", providers=["CPUExecutionProvider"])
        
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # TODO: store conf_thr and iou_thr on self (not used until Step 5)
        pass

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(640/h, 640/w)
        # TODO: new_w, new_h = round(w * scale), round(h * scale)
        new_w, new_h = round(w * scale), round(h * scale)
        # TODO: pad_x, pad_y = integer-divide the leftover space by 2   (use //)
        pad_x, pad_y = (640 - new_w) // 2, (640 - new_h) // 2

        resized_frame = cv2.resize(frame, (new_w, new_h))
        canvas = numpy.full((640,640,3),144, dtype=numpy.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_frame

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        float32 = rgb.astype(numpy.float32) / 255.0
        transposed = float32.transpose(2, 0, 1)
        # expand = numpy.expand_dims(x, axis=0)

        return canvas


if __name__ == "__main__":
    detector = OnnxDetector("learning/drone-model.onnx", ["CPUExecutionProvider"])

    cap = cv2.VideoCapture("learning/drone_vid.mp4")
    ok, frame = cap.read()
    cap.release()
    print(detector.preprocess(frame))
    # print(frame.shape)

# Piece 5: Turn it into the model's input
# Add the conversion chain from my first message: BGR to RGB, then float32, then / 255, then transpose, then add the batch axis, then make it contiguous. Change the return to the final version: return tensor, scale, pad_x, pad_y.

# Done when print(tensor.shape, tensor.dtype) shows (1, 3, 640, 640) float32.

# Piece 6: Feed it to the model
# In __main__:


# tensor, scale, pad_x, pad_y = detector.preprocess(frame)
# output = detector.session.run(None, {detector.input_name: tensor})
# print(output[0].shape)
# Done when it prints (1, 5, 8400). That means your preprocessing produced a valid input and the model ran on it. That's the end of Step 4.

# Do Piece 1 now. If you get stuck or see an error, paste the code and the output here. Once it prints images, move on to Piece 2.



