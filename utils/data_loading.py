import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask=True):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]

            img_ndarray = (img_ndarray - img_ndarray.min()) / (
                img_ndarray.max() - img_ndarray.min()
            )

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(filename))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)  # return (w, h)

    @staticmethod
    def transform(img, mask):
        data_transforms = A.Compose(
            [
                # A.Resize(height, width),
                A.SafeRotate(limit=[5, 15], p=0.5),
                A.Flip(),
                A.GridDistortion(p=0.5),
                A.ColorJitter(p=0.5),
                A.RandomSunFlare(p=0.2),
                A.RandomRain(p=0.1),
                A.RandomFog(p=0.1),
                A.RandomShadow(p=0.1),
                A.RandomSnow(p=0.1),
                A.MotionBlur(p=0.2),
                ToTensorV2(),
            ]
        )
        return data_transforms(image=img, mask=mask)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + ".*"))  # reg exp match
        img_file = list(self.images_dir.glob(name + ".*"))

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale)

        result = self.transform(img.astype("float32"), mask.astype("float32"))

        return {
            "image": result["image"].float().contiguous(),
            "mask": result["mask"].long().contiguous(),
        }
