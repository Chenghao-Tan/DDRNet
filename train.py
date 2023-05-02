import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from FP16 import evaluate
from models import DDRNet, UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.unified_focal_loss import AsymmetricUnifiedFocalLoss

dir_img = "./data/imgs/"
dir_mask = "./data/masks/"
dir_checkpoint = "./checkpoints/"


def train_net(
    net,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    val_percent: float = 0.1,
    save_checkpoint: bool = False,
    img_scale: float = 1.0,
    amp: bool = False,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="GSoC", resume="allow", anonymous="must")
    if experiment is not None:
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=val_percent,
                save_checkpoint=save_checkpoint,
                img_scale=img_scale,
                amp=amp,
            )
        )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    """
    optimizer = optim.SGD(
        net.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-8
    )
    """
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=math.ceil(n_train / batch_size),
        T_mult=2,
        eta_min=0,
        last_epoch=-1,
    )  # remain to be optimized TODO
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()  # TODO
    criterion2 = dice_loss(multi_class=True)  # TODO
    global_step = 0

    # 5. Begin training
    best = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                assert images.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device)
                one_hot_masks = F.one_hot(true_masks, num_classes=5).permute(0, 3, 1, 2)
                ignore_masks = one_hot_masks[:, 4, :, :].unsqueeze(dim=1)
                true_masks = one_hot_masks[:, 0, :, :].unsqueeze(dim=1)

                with torch.cuda.amp.autocast_mode.autocast(enabled=amp):
                    masks_pred = net(images)

                    ignore_area = True
                    if ignore_area:
                        masks_pred = torch.where(
                            ignore_masks.byte(),
                            torch.zeros_like(masks_pred),
                            masks_pred,
                        )

                    loss = criterion(masks_pred, true_masks.float()) + criterion2(
                        torch.sigmoid(masks_pred), true_masks
                    )  # TODO

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()  # step every batch here actually

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if experiment is not None:
                    experiment.log(
                        {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                    )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = n_train // (
                    2 * batch_size
                )  # the frequency of evaluation
                if division_step > 0 and global_step % division_step == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace("/", ".")
                        histograms[f"Weights/{tag}"] = wandb.Histogram(value.data.cpu())

                        histograms[f"Gradients/{tag}"] = wandb.Histogram(
                            value.grad.data.cpu()
                        )

                    val_pre, val_rec, val_miou = evaluate(net, val_loader, device)

                    logging.info(
                        "\n Validation Precision: {:4f} \n Validation Recall: {:.4f} \n Validation mIoU: {:.4f}".format(
                            val_pre, val_rec, val_miou
                        )
                    )
                    if experiment is not None:
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation Precision": val_pre,
                                "validation Recall": val_rec,
                                "validation mIoU": val_miou,
                                "images": wandb.Image(images[0].cpu()),
                                "masks": {
                                    "true": wandb.Image(true_masks[0].float().cpu()),
                                    "pred": wandb.Image(
                                        torch.softmax(masks_pred, dim=1)
                                        .argmax(dim=1)[0]
                                        .float()
                                        .cpu()
                                        if net.n_classes > 1
                                        else (
                                            torch.sigmoid(masks_pred)[0, 0, :, :]
                                            .float()
                                            .cpu()
                                            > 0.5
                                        ).float()
                                    ),
                                },
                                "step": global_step,
                                "epoch": epoch,
                                **histograms,
                            }
                        )
                    if val_miou > best:
                        best = val_miou
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        torch.save(
                            net.state_dict(),
                            f"{dir_checkpoint}BEST_mIoU{best}.pth",
                        )
                        logging.info(f"Best(mIoU{best}) pth saved!")

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(
                net.state_dict(), f"{dir_checkpoint}checkpoint_epoch{epoch + 1}.pth"
            )
            logging.info(f"Checkpoint {epoch + 1} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=127, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=3e-4,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load",
        "-f",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # net = UNet(n_channels=3, n_classes=1)
    net = DDRNet(n_channels=3, n_classes=1, pretrained="./DDRNet23s_imagenet.pth")

    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "./INTERRUPTED.pth")
        logging.info("Saved interrupt")
        sys.exit(0)
