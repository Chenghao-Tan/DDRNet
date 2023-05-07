import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tf

from models import DDRNet
from utils.dice_score import compute_pre_rec_miou


@torch.no_grad()
def predict_torch(model, input, output_dir, mask=None, confidence=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DDRNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load(model, map_location=device))
    net.half()  # FP16
    net.eval()
    net.to(device)
    print(f"Model: {model}, Device: {device}")
    image = tf.ToTensor()(Image.open(input)).half().to(device)  # FP16
    image = (image - image.min()) / (image.max() - image.min())
    image = image.unsqueeze(dim=0)
    mask_pred = net(image)
    mask_pred = (
        torch.softmax(mask_pred, dim=1)
        if net.n_classes > 1
        else torch.sigmoid(mask_pred)
    )
    if confidence is not None:
        mask_pred = (mask_pred > confidence).half()  # FP16
    output = os.path.join(
        output_dir, (os.path.splitext(os.path.split(input)[-1])[0] + "_pred.png")
    )
    tf.ToPILImage()((mask_pred).squeeze().cpu()).save(output)  # Auto mul(255)
    print(f"{input} -> {output}")
    if mask is not None:  # MaSTr1325 format
        mask = torch.as_tensor(np.array(Image.open(mask))).long().to(device).squeeze()
        mask = mask.unsqueeze(dim=0)
        one_hot_masks = F.one_hot(mask, num_classes=5).permute(0, 3, 1, 2)
        ignore_mask = one_hot_masks[:, 4, :, :].unsqueeze(dim=1)
        true_mask = one_hot_masks[:, 0, :, :].unsqueeze(dim=1)
        true_mask = true_mask.half()  # FP16
        mask_pred = torch.where(
            ignore_mask.bool(), torch.zeros_like(mask_pred), mask_pred
        )
        pre, rec, iou = compute_pre_rec_miou(mask_pred, true_mask, multi_class=True)
        print(f"Precision: {pre}, Recall: {rec}, IoU: {iou}")


@torch.no_grad()
def predict_onnx(model, input, output_dir, mask=None, confidence=None):
    net = ort.InferenceSession(model)
    print(f"Model: {model}")
    inputs = net.get_inputs()
    input_shape = inputs[0].shape[2:4]
    image = np.array(Image.open(input).resize((input_shape[1], input_shape[0])))
    image = image.transpose(2, 0, 1)  # hwc->chw
    image = (image - image.min()) / (image.max() - image.min())
    image = np.expand_dims(image, 0)  # batch dim
    rgb = image.astype(np.float32)
    if len(inputs) == 2:
        depth = np.ones((input_shape[1], input_shape[0])) * 255
        depth = np.stack((np.zeros_like(depth), depth), -1).reshape(
            1, 1, depth.shape[0], depth.shape[1] * 2
        )
        pred = net.run(None, {"rgb": rgb, "depth": depth})
        print(pred)
    else:  # len(inputs) == 1
        pred = net.run(None, {"rgb": rgb})[0]
        output = os.path.join(
            output_dir, (os.path.splitext(os.path.split(input)[-1])[0] + "_pred.png")
        )
        if confidence is not None:
            pred = (pred > confidence).astype(np.uint8)
        Image.fromarray((pred * 255).squeeze().astype(np.uint8)).save(output)
        print(f"{input} -> {output}")
        if mask is not None:  # MaSTr1325 format
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pred = torch.tensor(pred).to(device)
            mask = Image.open(mask).resize(
                (input_shape[1], input_shape[0]), resample=Image.NEAREST
            )
            mask = torch.as_tensor(np.array(mask)).long().to(device).squeeze()
            mask = mask.unsqueeze(dim=0)
            one_hot_masks = F.one_hot(mask, num_classes=5).permute(0, 3, 1, 2)
            ignore_mask = one_hot_masks[:, 4, :, :].unsqueeze(dim=1)
            true_mask = one_hot_masks[:, 0, :, :].unsqueeze(dim=1)
            pred = torch.where(ignore_mask.bool(), torch.zeros_like(pred), pred)
            pre, rec, iou = compute_pre_rec_miou(pred, true_mask, multi_class=True)
            print(f"Precision: {pre}, Recall: {rec}, IoU: {iou}")


if __name__ == "__main__":
    # predict_torch(model="BEST.pth", input="input.png", output_dir="./")
    predict_onnx(model="BEST.onnx", input="input.png", output_dir="./")
