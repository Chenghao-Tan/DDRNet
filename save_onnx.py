# IO Resolution
HEIGHT = 360  # TODO
WIDTH = 640  # TODO

# Built-in Obstacle Detection
DETECTION = True  # TODO

# Built-in Obstacle Detection
# Segmentation Confidence Threshold
CONFIDENCE = 0.9  # TODO

# Built-in Obstacle Detection
# Grid Settings
GRID_NUM_H = 5  # TODO
GRID_NUM_W = 5  # TODO
THRESHOLD = 0.2  # TODO


assert HEIGHT % GRID_NUM_H == 0
assert WIDTH % GRID_NUM_W == 0


import torch

from models import DDRNet23s, UNet


def export_onnx(
    model,
    output,
    height,
    width,
    detection,
    confidence,
    grid_height,
    grid_width,
    grid_threshold,
):
    net = DDRNet23s(n_channels=3, n_classes=1, scale_factor=8)
    net.load_state_dict(torch.load(model))
    net.eval()
    net.pre_process.enable(True)
    net.post_process.enable(True)
    net.post_process.set_grids(
        confidence=confidence,
        grid_height=grid_height,
        grid_width=grid_width,
        threshold=grid_threshold,
    )  # type: ignore
    dummy_rgb = torch.randn(1, 3, height, width)
    dummy_depth = torch.randn(1, 1, height, width * 2) if detection else None
    torch.onnx.export(
        net,
        (dummy_rgb, dummy_depth),
        output,
        input_names=["rgb", "depth"],
        output_names=["out"],
        # output_names=["out", "debug"],  # For debug
        opset_version=11,
    )


if __name__ == "__main__":
    export_onnx(
        model="./BEST.pth",
        output="./BEST.onnx",
        height=HEIGHT,
        width=WIDTH,
        detection=DETECTION,
        confidence=CONFIDENCE,
        grid_height=HEIGHT // GRID_NUM_H,
        grid_width=WIDTH // GRID_NUM_W,
        grid_threshold=THRESHOLD,
    )
