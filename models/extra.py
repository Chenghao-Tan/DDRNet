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
            if mask.shape[1] == 1:  # channel_dim
                mask = torch.sigmoid(mask)
            else:
                mask = torch.nn.functional.softmax(mask, dim=1)  # type: ignore

            if depth is not None:
                grid_height = 36  # TODO
                grid_width = 64  # TODO
                threshold = 0.2  # TODO

                assert (
                    mask.shape[2] % grid_height == 0 and mask.shape[3] % grid_width == 0
                )
                grid_num_h = mask.shape[2] // grid_height
                grid_num_w = mask.shape[3] // grid_width
                if isinstance(threshold, float):
                    assert threshold >= 0 and threshold <= 1
                    threshold = threshold * grid_height * grid_width

                mask = mask > 0.5
                depth /= 1000  # mm->m
                # depth = torch.nn.functional.avg_pool2d(depth, kernel_size=8, stride=8)  # type: ignore # TODO
                # depth[depth < 0.35] = 0  # TODO
                # depth[depth > 35] = 0  # TODO
                filtered = torch.mul(mask, depth)
                grids = (
                    filtered.reshape(grid_num_h, grid_height, grid_num_w, grid_width)
                    .permute(0, 2, 1, 3)
                    .reshape(grid_num_h * grid_num_w, grid_height * grid_width)
                )
                non_zero_num = torch.where(
                    grids > 0, torch.ones_like(grids), torch.zeros_like(grids)
                ).sum(dim=1)
                non_zero_num[non_zero_num == 0] = -1  # In case of getting nan
                z = (
                    grids.sum(dim=1) / non_zero_num
                )  # To get the mean value of the nonzeros

                label = torch.where(
                    non_zero_num > threshold,
                    torch.ones_like(non_zero_num),
                    torch.zeros_like(non_zero_num),
                )  # 0:background 1:obstacle

                out = torch.stack((label, z), dim=1)
                # return out.flatten(), filtered  # For debug
                return out.flatten()
            else:
                return mask
