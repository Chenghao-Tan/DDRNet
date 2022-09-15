# IO Resolution
HEIGHT = 360  # TODO
WIDTH = 640  # TODO

# Built-in Obstacle Detection
DETECTION = True

# Segmentation Confidence Threshold
CONFIDENCE = 0.9  # TODO

# Grid Settings
GRID_HEIGHT = 72  # TODO
GRID_WIDTH = 128  # TODO
THRESHOLD = 0.2  # TODO


assert HEIGHT % GRID_HEIGHT == 0
assert WIDTH % GRID_WIDTH == 0


import torch

from models import DDRNet23s, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DDRNet23s(n_channels=3, n_classes=1, scale_factor=8)

net.load_state_dict(torch.load("./BEST.pth"))
net.eval()
net.pre_process.enable(True)
net.post_process.enable(True)
net.post_process.set_grids(
    confidence=CONFIDENCE,
    grid_height=GRID_HEIGHT,
    grid_width=GRID_WIDTH,
    threshold=THRESHOLD,
)  # type: ignore
dummy_rgb = torch.randn(1, 3, HEIGHT, WIDTH)
dummy_depth = torch.randn(1, 1, HEIGHT, WIDTH * 2) if DETECTION else None
torch.onnx.export(
    net,
    (dummy_rgb, dummy_depth),
    "./BEST.onnx",
    input_names=["rgb", "depth"],
    output_names=["out"],
    # output_names=["out", "debug"],  # For debug
    opset_version=11,
)
