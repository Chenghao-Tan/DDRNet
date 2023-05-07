import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import compute_pre_rec_miou


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    precision = 0
    recall = 0
    miou = 0

    # iterate over the validation set
    for batch in tqdm(
        dataloader,
        total=num_val_batches,
        desc="Validation round",
        unit="batch",
        leave=False,
        disable=None,
    ):
        image, true_masks = batch["image"], batch["mask"]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device)
        one_hot_masks = F.one_hot(true_masks, num_classes=5).permute(0, 3, 1, 2)
        ignore_masks = one_hot_masks[:, 4, :, :].unsqueeze(dim=1)
        true_masks = one_hot_masks[:, 0, :, :].unsqueeze(dim=1)

        with torch.no_grad():
            # predict the mask
            masks_pred = net(image)
            masks_pred = (
                torch.softmax(masks_pred, dim=1)
                if net.n_classes > 1
                else torch.sigmoid(masks_pred)
            )

            ignore_area = True
            if ignore_area:
                masks_pred = torch.where(
                    ignore_masks.bool(), torch.zeros_like(masks_pred), masks_pred
                )

            # compute the Precision and the Recall
            pre, rec, iou = compute_pre_rec_miou(
                masks_pred, true_masks, multi_class=True
            )
            precision += pre
            recall += rec
            miou += iou

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return precision, recall, miou
    else:
        return (
            precision / num_val_batches,
            recall / num_val_batches,
            miou / num_val_batches,
        )
