import torch
from ptflops import get_model_complexity_info

from models import DDRNet, UNet

with torch.cuda.device(0):
    net = DDRNet(n_channels=3, n_classes=1)
    macs, params = get_model_complexity_info(
        net, (3, 360, 640), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
