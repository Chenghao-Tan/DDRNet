import torch

from models import DDRNet23s, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DDRNet23s(n_channels=3, n_classes=1, scale_factor=8)

net.load_state_dict(torch.load("./BEST.pth"))
net.eval()
net.extra_process(True)
dummy_rgb = torch.randn(1, 3, 360, 640)
dummy_depth = torch.randn(1, 1, 360, 1280)  # U16->U8
torch.onnx.export(
    net,
    (dummy_rgb, dummy_depth),
    "./BEST.onnx",
    input_names=["rgb", "depth"],
    output_names=["out"],
    opset_version=11,
)
