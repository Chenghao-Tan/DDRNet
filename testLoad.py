import numpy as np
from PIL import Image

I = Image.open("/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/imgs/0.png")
I_array = np.array(I)
I_array = np.expand_dims(I_array, 0)  # channel dim
I_array = np.expand_dims(I_array, 0)  # batch dim
I_array = I_array.astype(np.float32)
print(I_array.shape)

import onnxruntime as ort

ort_session = ort.InferenceSession(
    "/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/unet_model.onnx"
)
ort_inputs = {ort_session.get_inputs()[0].name: I_array}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)
print(ort_outs[0] * 255)
Image.fromarray((ort_outs[0] * 255).astype(np.byte)).save("./onnx_out.png")
