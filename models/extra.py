import torch


class pre(torch.nn.Module):
    def __init__(self):
        super(pre, self).__init__()
        self.en = False

    def forward(self, rgb, depth=None):
        with torch.no_grad():
            if self.en:
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

                if depth is not None:
                    depth = (
                        256.0 * depth[:, :, :, 1::2] + depth[:, :, :, ::2]
                    )  # U8 to FP(16)

        return rgb, depth

    def enable(self, en):
        self.en = en


class post(torch.nn.Module):
    def __init__(self):
        super(post, self).__init__()
        self.en = False
        self.confidence = 0.9
        self.grid_height = 72
        self.grid_width = 128
        self.threshold = 0.2

    def forward(self, mask, depth=None):
        with torch.no_grad():
            if self.en:
                if mask.shape[1] == 1:  # channel_dim
                    mask = torch.sigmoid(mask)
                else:
                    mask = torch.nn.functional.softmax(mask, dim=1)  # type: ignore

                if depth is not None:
                    assert (
                        mask.shape[2] % self.grid_height == 0
                        and mask.shape[3] % self.grid_width == 0
                    )
                    grid_num_h = mask.shape[2] // self.grid_height
                    grid_num_w = mask.shape[3] // self.grid_width
                    if isinstance(self.threshold, float):
                        assert self.threshold >= 0 and self.threshold <= 1
                        self.threshold = (
                            self.threshold * self.grid_height * self.grid_width
                        )

                    assert self.confidence >= 0 and self.confidence <= 1
                    mask = mask > self.confidence
                    depth /= 1000  # mm->m

                    # You may want to downscale high resolution depth map to get more stable values
                    # depth = torch.nn.functional.avg_pool2d(depth, kernel_size=8, stride=8)  # type: ignore

                    filtered = torch.mul(mask, depth)
                    grids = (
                        filtered.reshape(
                            grid_num_h, self.grid_height, grid_num_w, self.grid_width
                        )
                        .permute(0, 2, 1, 3)
                        .reshape(
                            grid_num_h * grid_num_w, self.grid_height * self.grid_width
                        )
                    )
                    non_zero_num = torch.where(
                        grids > 0, torch.ones_like(grids), torch.zeros_like(grids)
                    ).sum(dim=1)
                    non_zero_num[non_zero_num == 0] = -1  # In case of getting nan
                    z = (
                        grids.sum(dim=1) / non_zero_num
                    )  # To get the mean value of the nonzeros

                    label = torch.where(
                        non_zero_num > self.threshold,
                        torch.ones_like(non_zero_num),
                        torch.zeros_like(non_zero_num),
                    )  # 0:background 1:obstacle

                    out = torch.stack((label, z), dim=1)
                    # return out.flatten(), filtered  # For debug
                    return out.flatten()

        return mask

    def enable(self, en):
        self.en = en

    def set_grids(self, confidence, grid_height, grid_width, threshold):
        self.confidence = confidence
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.threshold = threshold
