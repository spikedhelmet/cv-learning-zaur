import onnxruntime as ort

session = ort.InferenceSession("learning/drone-model.onnx", providers=["CPUExecutionProvider"])

model_input = session.get_inputs()[0]
print(f"Input Name:  {model_input.name}")
print(f"Input Shape: {model_input.shape}")
print(f"Input Type:  {model_input.type}")

model_output = session.get_outputs()[0]
print(f"Output Name:  {model_output.name}")
print(f"Output Shape: {model_output.shape}")
print(f"Output Type:  {model_output.type}")
