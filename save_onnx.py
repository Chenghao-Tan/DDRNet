import torch

from models import DDRNet, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DDRNet(n_channels=3, n_classes=1)

net.load_state_dict(torch.load("./BEST.pth"))
net.eval()
net.extra_process(True)
dummy_input = torch.randn(1, 3, 360, 640)
torch.onnx.export(
    net, dummy_input, "./BEST.onnx", opset_version=11,
)
