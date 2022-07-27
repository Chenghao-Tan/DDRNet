import numpy as np
from PIL import Image

I = Image.open("./input.png").resize((640, 360))
rgb = np.array(I)
if len(rgb.shape) == 2:
    rgb = np.expand_dims(rgb, 0)  # channel dim
else:
    rgb = rgb.transpose(2, 0, 1)  # hwc->chw
rgb = np.expand_dims(rgb, 0)  # batch dim
rgb = rgb.astype(np.float32)
print(rgb.shape)

# simulate U8 depth
depth = np.ones((360, 640)) * 255
depth = np.stack((np.zeros_like(depth), depth), -1).reshape(
    1, 1, depth.shape[0], depth.shape[1] * 2
)
print(depth.shape)


import onnxruntime as ort

ort_session = ort.InferenceSession("./BEST.onnx")
ort_inputs = {"rgb": rgb, "depth": depth}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)
print(ort_outs[0] * 255)
Image.fromarray((ort_outs[0] * 255).astype(np.byte)).save("./output.png")
