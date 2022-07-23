import torch

from models import DDRNet, UNet


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DDRNet(n_channels=3, n_classes=1)

net.load_state_dict(torch.load("./BEST.pth"))
net.eval()
net.extra_process(True)
# net.half()
dummy_rgb = torch.randn(1, 3, 360, 640)
dummy_depth = torch.randn(1, 1, 360, 1280)  # U16->U8
# dummy_input.half()
torch.onnx.export(
    net,
    (dummy_rgb, dummy_depth),
    "./BEST.onnx",
    input_names=["rgb", "depth"],
    output_names=["out"],
    opset_version=11,
)
print(get_parameter_number(net))
