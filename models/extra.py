import torch


class pre(torch.nn.Module):
    def __init__(self):
        super(pre, self).__init__()

    def forward(self, rgb, depth):
        with torch.no_grad():
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

            if depth is not None:
                depth = (
                    256.0 * depth[:, :, :, 1::2] + depth[:, :, :, ::2]
                )  # U8 to FP(16)

        return rgb, depth


class post(torch.nn.Module):
    def __init__(self):
        super(post, self).__init__()

    def forward(self, mask, depth):
        with torch.no_grad():
            if mask.shape[1] == 1:
                mask = torch.sigmoid(mask)
            else:
                mask = torch.nn.functional.softmax(mask, dim=1)  # type: ignore

            if depth is not None:  # TODO
                depth = torch.nn.functional.avg_pool2d(depth, kernel_size=8, stride=8)  # type: ignore
                mask = torch.nn.functional.avg_pool2d(mask, kernel_size=8, stride=8) > 0.5  # type: ignore
                out = torch.mul(mask, depth)
                out, _ = torch.median(out, dim=2)
            else:
                out = mask

        return out
