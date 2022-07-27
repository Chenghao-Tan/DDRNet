import torch
from thop import profile

from models import DDRNet23s, UNet

with torch.cuda.device(0):
    net = DDRNet23s(n_channels=3, n_classes=1, scale_factor=8)
    net.eval()
    net.extra_process(True)
    dummy_rgb = torch.randn(1, 3, 360, 640)
    dummy_depth = torch.randn(1, 1, 360, 1280)  # U16->U8
    macs, params = profile(net, inputs=(dummy_rgb, dummy_depth))
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
