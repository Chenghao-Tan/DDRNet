import torch
import torch.nn as nn
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    input = input.float()
    target = target.float()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)
    if sets_sum.item() == 0:
        return torch.tensor(1, dtype=torch.float32).to(input.device)
    else:
        return (2 * inter + epsilon) / (sets_sum + epsilon)


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    dice = torch.tensor(0, dtype=torch.float32).to(input.device)
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], epsilon)
    return dice / input.shape[1]


class dice_loss(nn.Module):
    def __init__(self, multi_class=False):
        super().__init__()
        if multi_class:
            self.compute_dice = dice_coeff
        else:
            self.compute_dice = multiclass_dice_coeff

    def forward(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        return 1 - self.compute_dice(input, target)


def compute_pre(input: Tensor, target: Tensor):
    input = input.float()
    target = target.float()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(input)
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def compute_rec(input: Tensor, target: Tensor):
    input = input.float()
    target = target.float()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(target)
    if total.item() == 0:
        return 1
    result = inter.item() / total.item()
    return result


def compute_iou(input: Tensor, target: Tensor):
    input = input.float()
    target = target.float()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    total = torch.sum(input) + torch.sum(target) - inter
    if total.item() == 0:
        return 0
    result = inter.item() / total.item()
    return result


def multiclass_metrics(input, target, func):
    assert input.size() == target.size()
    sum = 0
    for channel in range(input.shape[1]):
        sum += func(input[:, channel, ...], target[:, channel, ...])
    return sum / input.shape[1]


def compute_pre_rec_miou(input: Tensor, target: Tensor, multi_class: bool = True):
    assert input.size() == target.size()
    if multi_class:
        assert input.shape[1] == target.shape[1]
        pre = rec = miou = 0.0
        for c in range(input.shape[1]):
            pre += compute_pre(input[:, c, ...], target[:, c, ...])
            rec += compute_rec(input[:, c, ...], target[:, c, ...])
            miou += compute_iou(input[:, c, ...], target[:, c, ...])
        pre /= input.shape[1]
        rec /= input.shape[1]
        miou /= input.shape[1]
        return pre, rec, miou
    else:
        return (
            compute_pre(input, target),
            compute_rec(input, target),
            compute_iou(input, target),
        )
