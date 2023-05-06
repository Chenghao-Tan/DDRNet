# DDRNet for Boat Obstacle Avoidance
**This project is modified from [U-Net: Semantic segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet)**

**This branch (master) is used with the latest [Boat-Obstacle-Avoidance](https://github.com/Chenghao-Tan/Boat-Obstacle-Avoidance). Note that you can now generate a dataset from a set of photos taken from the on-boat camera's perspective to improve the model's performance in specific areas. See Use SAM.**

**The .pth weights file is generic, changing onnx export settings or switching between DDRNet branches does not require retraining, just use the same .pth and rerun `save_onnx.py`.**


## Overview
This is a custom training framework supporting **DDRNet and UNet(deprecated).** It is used to train the AI model required by [Boat-Obstacle-Avoidance](https://github.com/Chenghao-Tan/Boat-Obstacle-Avoidance). (Prepare your own dataset, train a new model, export the model in ONNX format, then use [Luxonis blobconverter](http://blobconverter.luxonis.com/) to convert the ONNX to blob, which is required by the scripts.)

The built-in models are modified and **use MaSTr1325 format dataset**. See **Data format**.

You can export onnx file for [Luxonis blobconverter](http://blobconverter.luxonis.com/). See **Export ONNX**.

- Number of parameters: **5.7M**
- Computational complexity: **4.1G** (automatically generated value, may be extremely inaccurate)


## GUI
A web-based GUI is provided. Complete the preparation steps in **Prepare** and **Use SAM**, then enjoy!
```bash
python webui.py
```


## Prepare
*Note: Use Python 3.8 or newer*

**First, install PyTorch(>=1.7) and torchvision(>=0.8) with CUDA support.** It's recommended to use **conda**. Please follow the official instructions [here](https://pytorch.org/get-started/locally/). Then install the rest packages (except SAM) through **pip**:

```bash
pip install -r requirements.txt
```


## Use SAM (optional)
You can use SAM (Segment-Anything-Model) to generate your unique dataset automatically. This step will significantly improve the performance of the model in specific areas. You are also welcome to submit your dataset to the community for building better models in the future.

```bash
git submodule update --init --recursive
cd segment-anything
pip install -e .
```

Download the checkpoint(s) from Meta. See [here](https://github.com/facebookresearch/segment-anything#model-checkpoints). Take as many pictures as possible from the on-boat camera's perspective. Then you can start dataset generation without manually annotating:

```console
> python SAM.py -h
usage: SAM.py [-h] -m TYPE -l PATH -s PATH [-e EXT] [-t PATH] [-b SIZE] [-n I O] [-a LEVEL] [-o W H] [--no-multimask] [--visualize]

Use SAM to automatically generate dataset (MaSTr1325-like).

options:
  -h, --help            show this help message and exit
  -m TYPE, --model TYPE
                        vit_h/vit_l/vit_b
  -l PATH, --load PATH  Load model from a .pth file
  -s PATH, --source PATH
                        Unlabeled image source (non-recursive)
  -e EXT, --ext EXT     Filter input image by extension (png/jpg/...)
  -t PATH, --target PATH
                        Output location
  -b SIZE, --batch-size SIZE
                        Batch size (increase = more VRAM & more speed)
  -n I O, --num-workers I O
                        Number of processes loading and writing data, respectively
  -a LEVEL, --annotation-level LEVEL
                        1->water, 2->water&sky
  -o W H, --output-size W H
                        Size of the output images and masks (WxH) (0 0 = unchanged)
  --no-multimask        Generate a single mask instead of picking the best one
  --visualize           Visualize the output mask (for debug)
```

### Tips:
- You may have to wait a while for it to load. Specify `--model`, `--load` and `--source`, and then it will automatically annotate the images and place them in the training folder (`./data`).
- Performance: **vit_h** > **vit_l** > **vit_b**. Speed is in reverse. When `--batch-size` is set to 1, **vit_b** needs < 4GB VRAM, **vit_l** needs < 6GB, **vit_h** needs < 8GB. More `--batch-size` usually means more speed, but it also takes more VRAM and needs a more powerful GPU.
- Please set `--annotation-level` according to the pictures you take. For example, if there are mostly water and obstacles, barely sky or other things, set it to 1. This is because the prompt points are fixed at 10% below and above the edge of the image.


## Training
```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

### Tips:
- Recommend to set `--epochs` to 1,3,7,15,31,63...
- You can use `--scale` to scale the image while keeping the aspect ratio, but it's recommended to modify [data_loading.py](https://github.com/Chenghao-Tan/DDRNet/blob/master/utils/data_loading.py) directly. (Uncomment **A.Resize(height, width)**).
- `--amp` is not recommended as it may drastically reduce precision.

Trainable parameters will be saved to the `checkpoints` folder as .pth. Only the best one will be saved by default.


## Export ONNX (for blob converting)
You can run [save_onnx.py](https://github.com/Chenghao-Tan/DDRNet/blob/master/save_onnx.py) to convert `"./BEST.pth"` to `"./BEST.onnx"`. You can change **IO resolution and detection settings** of the onnx model to be exported in this file. You can turn off obstacle detection (to hand it over to the companion computer) so that the model only outputs a confidence map.

*Note: the height and width of the grid must be divisible by the resolution.*

Check **# For debug** tag in both [save_onnx.py](https://github.com/Chenghao-Tan/DDRNet/blob/master/save_onnx.py) and [models/extra.py](https://github.com/Chenghao-Tan/DDRNet/blob/master/models/extra.py) for how to export the obstacle detection debug version.

You can convert the **ONNX** to **blob** [here](https://blobconverter.luxonis.com/), to use it in [Boat-Obstacle-Avoidance](https://github.com/Chenghao-Tan/Boat-Obstacle-Avoidance). (**Complete steps: .pth->.onnx->.blob**)

Useful information:
- Model's IO:
  - input->Image ("rgb"), Depth map ("depth")
  - no detection input->Image ("rgb")
  - output->Flattened grids info ("out")
  - debug output->Flattened grids info ("out"), Filtered depth map ("debug")
  - no detection output->confidence map ("out")
- Flattened grids info: (label, z for each grid, flattened in 1D)
  - label: 0 for background, 1 for obstacles (binary classification)
  - z: depth, in meters


## Other tools
- [flops.py](https://github.com/Chenghao-Tan/DDRNet/blob/master/flops.py) is for getting the model's number of parameters and estimated computational complexity.


## Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks, are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
You can load your own pth as pretrained value using `--load`. (The DDRNet model will load `DDRNet23s_imagenet.pth` first by default whether using `--load` or not)

There's also a model (`mIoU_0.9042.pth`) pretrained on MaSTr1325 dataset. (mIoU 0.9042, trained & validated at 640*360)


## Data format
The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). Images should be **jpg** and masks should be **png**. Images and masks should have the same name and resolution.

As MaSTr1325 does, labels in masks should correspond to the following values:
  - Obstacles and environment = 0 (value zero)
  - Water = 1 (value one)
  - Sky = 2 (value two)
  - Ignore region / unknown category = 4 (value four)

However, the framework only does binary classification by default (ignore region is still used).  Therefore in the DDRNet output label, 1 represents the obstacle and 0 represents the background.
