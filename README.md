# DDRNet for Boat Obstacle Avoidance
**This project is modified from [U-Net: Semantic segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet)**

**Check [master branch](https://github.com/Agent-Birkhoff/DDRNet) if you want to train your own model for the \*LATEST\* [Boat-Obstacle-Avoidance](https://github.com/Agent-Birkhoff/Boat-Obstacle-Avoidance).**


## Overview
This is a custom training framework supporting **DDRNet and UNet(deprecated).**

The built-in models are modified and **use MaSTr1325 format dataset**. See **Data**. You can make the trained model end-to-end by opening the extra process option. See **Export ONNX**.

- Number of parameters: **5.7M**
- Computational complexity: **4.1G** (automatically generated value, may be extremely inaccurate)


## Prepare
*Note: Use Python 3.6 or newer*

```bash
pip install -r requirements.txt
```


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
- You can use `--scale` to scale the image while keeping the aspect ratio, but it's recommended to modify [data_loading.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/utils/data_loading.py) directly. (Uncomment **A.Resize(height, width)**).
- `--amp` is not recommended as it may drastically reduce precision.

Trainable parameters will be saved to the `checkpoints` folder in .pth. Only the best results will be saved by default. However, because the mIoU values appended to the filename are usually different, there could be more than one file.


## Export ONNX (for blob converting)
You can run [save_onnx.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/save_onnx.py) to convert `"./BEST.pth"` to `"./BEST.onnx"`. You can change the IO resolution in this file.

See **# For debug** tag in both [save_onnx.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/save_onnx.py) and [models/extra.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/models/extra.py) for how to export the debug version.

With **net.extra_process(True)**, the exported onnx model is end-to-end.
- Model's IO:
  - input->Image ("rgb"), Depth map ("depth")
  - output->Flattened grids info ("out")
  - debug output->Flattened grids info ("out"), Filtered depth map ("debug")
- Flattened grids info: (label, x, y, z for each grid, flattened in 1D)
  - label: 0 for background, 1 for obstacles (binary classification)
  - x,y,z: in meters


## Other tools
- [test_onnx.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/test_onnx.py) is for testing the exported onnx using onnxruntime. The default image input is `"./input.png"` and the depth map input is simulated with ones.
- [flops.py](https://github.com/Agent-Birkhoff/DDRNet/blob/master/flops.py) is for getting the model's number of parameters and estimated computational complexity.


## Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks, are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
You can load your own pth as pretrained value using `--load`. (The DDRNet model will load `DDRNet23s_imagenet.pth` first by default whether using `--load` or not)


## Data
The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). Images should be **jpg** and masks should be **png**. Images and masks should have the same name and resolution.

As MaSTr1325 does, labels in masks should correspond to the following values:
  - Obstacles and environment = 0 (value zero)
  - Water = 1 (value one)
  - Sky = 2 (value two)
  - Ignore region / unknown category = 4 (value four)

However, the framework only does binary classification by default (ignore region is still used).  Therefore in the output label, 1 represents the obstacle and 0 represents the background.
