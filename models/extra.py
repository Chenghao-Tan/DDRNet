import numpy as np
import torch


def z2xy_coefficient(
    grid_height,
    grid_width,
    grid_num_h,
    grid_num_w,
    intrinsic_matrix=np.array(
        [
            [492.23822021, 0.0, 320.57855225],
            [0.0, 492.23822021, 181.67709351],
            [0.0, 0.0, 1.0],
        ]
    ),  # TODO
):
    # Center of each grid
    grid_cx = np.linspace(
        (grid_width - 1) / 2,
        grid_width * grid_num_w - 1 - (grid_width - 1) / 2,
        grid_num_w,
    )
    grid_cy = np.linspace(
        (grid_height - 1) / 2,
        grid_height * grid_num_h - 1 - (grid_height - 1) / 2,
        grid_num_h,
    )
    grid_cx, grid_cy = np.meshgrid(grid_cx, grid_cy)  # To 2D mesh

    # Focal length
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    # Center of the image
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Projection coefficient
    x_coefficient = ((grid_cx - cx) / fx).flatten()
    y_coefficient = ((cy - grid_cy) / fy).flatten()

    return torch.tensor(x_coefficient), torch.tensor(y_coefficient)


def z2xy_coefficient_fov(grid_height, grid_width, grid_num_h, grid_num_w, hfov=69):
    # WARNING: Use HFOV only (assuming fx==fy)

    # Center of each grid
    grid_cx = np.linspace(
        (grid_width - 1) / 2,
        grid_width * grid_num_w - 1 - (grid_width - 1) / 2,
        grid_num_w,
    )
    grid_cy = np.linspace(
        (grid_height - 1) / 2,
        grid_height * grid_num_h - 1 - (grid_height - 1) / 2,
        grid_num_h,
    )
    grid_cx, grid_cy = np.meshgrid(grid_cx, grid_cy)  # To 2D mesh

    # H&W of the image
    height = grid_height * grid_num_h
    width = grid_width * grid_num_w
    # Assuming fx==fy
    fx_r = np.tan(hfov / 180 * np.pi / 2) / (width / 2)  # Reciprocal of fx
    fy_r = fx_r  # Reciprocal of fy
    # Center of the image
    cx = width / 2
    cy = height / 2

    # Projection coefficient
    x_coefficient = ((grid_cx - cx) * fx_r).flatten()
    y_coefficient = ((cy - grid_cy) * fy_r).flatten()

    return torch.tensor(x_coefficient), torch.tensor(y_coefficient)


class pre(torch.nn.Module):
    def __init__(self):
        super(pre, self).__init__()
        self.en = False

    def forward(self, rgb, depth):
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

    def forward(self, mask, depth):
        with torch.no_grad():
            if self.en:
                if mask.shape[1] == 1:  # channel_dim
                    mask = torch.sigmoid(mask)
                else:
                    mask = torch.nn.functional.softmax(mask, dim=1)  # type: ignore

                if depth is not None:
                    confidence = 0.9  # TODO
                    assert confidence >= 0 and confidence <= 1
                    grid_height = 72  # TODO
                    grid_width = 128  # TODO
                    threshold = 0.2  # TODO

                    assert (
                        mask.shape[2] % grid_height == 0
                        and mask.shape[3] % grid_width == 0
                    )
                    grid_num_h = mask.shape[2] // grid_height
                    grid_num_w = mask.shape[3] // grid_width
                    if isinstance(threshold, float):
                        assert threshold >= 0 and threshold <= 1
                        threshold = threshold * grid_height * grid_width

                    mask = mask > confidence
                    depth /= 1000  # mm->m
                    # depth = torch.nn.functional.avg_pool2d(depth, kernel_size=8, stride=8)  # type: ignore # TODO
                    # depth[depth < 0.35] = 0  # TODO
                    # depth[depth > 35] = 0  # TODO
                    filtered = torch.mul(mask, depth)
                    grids = (
                        filtered.reshape(
                            grid_num_h, grid_height, grid_num_w, grid_width
                        )
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

                    z2x, z2y = z2xy_coefficient(
                        grid_height, grid_width, int(grid_num_h), int(grid_num_w),
                    )  # TODO
                    x = z.mul(z2x.type_as(z))
                    y = z.mul(z2y.type_as(z))

                    out = torch.stack((label, x, y, z), dim=1)
                    # return out.flatten(), filtered  # For debug
                    return out.flatten()

        return mask

    def enable(self, en):
        self.en = en
