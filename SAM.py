import argparse
import logging
import os
from multiprocessing import Process, Queue
from shutil import copyfile

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision import transforms as tf
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Use SAM to automatically generate dataset (MaSTr1325-like)."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        required=True,
        help="vit_h/vit_l/vit_b",
        metavar="TYPE",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        required=True,
        help="Load model from a .pth file",
        metavar="PATH",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Unlabeled image source (non-recursive)",
        metavar="PATH",
    )
    parser.add_argument(
        "-e",
        "--ext",
        default=None,
        type=str,
        help="Filter input image by extension (png/jpg/...)",
        metavar="EXT",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="./data",
        type=str,
        help="Output location",
        metavar="PATH",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Batch size (increase = more VRAM & more speed)",
        metavar="SIZE",
        dest="batch_size",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        nargs=2,
        default=(1, 1),
        type=int,
        help="Number of processes loading and writing data, respectively",
        metavar=("I", "O"),
        dest="num_workers",
    )
    parser.add_argument(
        "-a",
        "--annotation-level",
        default=2,
        type=int,
        choices=[1, 2],
        help="1->water, 2->water&sky",
        metavar="LEVEL",
        dest="annotation_level",
    )
    parser.add_argument(
        "-o",
        "--output-size",
        nargs=2,
        default=(640, 360),
        type=int,
        help="Size of the output images and masks (WxH)",
        metavar=("W", "H"),
        dest="output_size",
    )
    parser.add_argument(
        "--no-multimask",
        action="store_false",
        default=True,  # Multimask==True
        help="Generate a single mask instead of picking the best one",
        dest="multimask",
    )
    return parser.parse_args()


def loading_worker(lQ, ids, transform, args):
    for id in ids:
        name = os.path.basename(id)
        image = Image.open(id)
        image_size = (image.size[1], image.size[0])  # H x W

        image = np.array(image)
        image = transform.apply_image(image)  # Recommend this over *_torch version
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        prompt_points = torch.tensor(  # Point(s) for water
            [
                [
                    [
                        image_size[1] // 2,
                        int(image_size[0] * 0.9),
                    ],  # 10% above bottom, horizontal middle
                    [
                        image_size[1] // 4,
                        int(image_size[0] * 0.9),
                    ],  # 10% above bottom, 25% from left
                    [
                        image_size[1] // 4 * 3,
                        int(image_size[0] * 0.9),
                    ],  # 10% above bottom, 25% from right
                ]
            ]
        )
        if args.annotation_level == 2:  # Add point(s) for sky
            prompt_points = torch.cat(
                [
                    prompt_points,
                    torch.tensor(
                        [
                            [
                                [
                                    image_size[1] // 2,
                                    int(image_size[0] * 0.1),
                                ],  # 10% below top, horizontal middle
                                [
                                    image_size[1] // 4,
                                    int(image_size[0] * 0.1),
                                ],  # 10% below top, 25% from left
                                [
                                    image_size[1] // 4 * 3,
                                    int(image_size[0] * 0.1),
                                ],  # 10% below top, 25% from right
                            ]
                        ]
                    ),
                ]
            )
        prompt_points = transform.apply_coords_torch(
            prompt_points, original_size=image_size
        )

        prompt_label = torch.tensor([[1, 1, 1]])  # Label(s) for water
        if args.annotation_level == 2:  # Add label(s) for sky
            prompt_label = torch.cat([prompt_label, torch.tensor([[1, 1, 1]])])

        lQ.put(
            {
                "image": image,
                "original_size": image_size,
                "point_coords": prompt_points,
                "point_labels": prompt_label,
                "name": name,
            },
        )


def writing_worker(wQ, args):
    while True:
        output_dict = wQ.get()
        name = output_dict["name"]
        masks = output_dict["masks"]
        scores = output_dict["iou_predictions"]

        if masks.shape[1] > 1:  # A.K.A. args.multimask==True
            masks = masks[
                torch.arange(masks.shape[0]), scores.argmax(dim=1)
            ]  # Select masks with the highest scores
        else:
            masks = masks[:, 0, :, :]  # A.K.A. masks.squeeze(dim=1)

        mask = masks[0].type(torch.uint8)  # 1->water
        for i, m in enumerate(masks[1:], start=2):
            mask += m * i  # 2->sky, 3->more...
        mask[mask > len(masks)] = 4  # Overlap defaults to 4 (unknown)

        threshold = 0.25  # Proportion of water&sky&... should be greater than threshold
        if ((mask > 0) & (mask != 4)).sum() / (
            mask.shape[0] * mask.shape[1]
        ) < threshold:
            logging.warning(
                f"Jumped one image ({name}) due to poor annotation quality."
            )
            continue

        mask = tf.ToPILImage()(mask)
        mask = mask.resize(args.output_size, resample=Image.NEAREST)  # WxH

        imgs_path = os.path.join(os.path.abspath(args.target), "imgs")
        masks_path = os.path.join(os.path.abspath(args.target), "masks")
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)
            logging.warning(f"Creating directory: {imgs_path}...")
        if not os.path.exists(masks_path):
            os.makedirs(masks_path)
            logging.warning(f"Creating directory: {masks_path}...")
        copyfile(
            os.path.join(os.path.abspath(args.source), name),
            os.path.join(imgs_path, f"{name}.png"),
        )  # Copy image
        mask.save(
            os.path.join(masks_path, f"{name}.png"),
            quality=100,  # In case of jpg
        )  # Save mask


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = "cuda"  # It will be painfully slow on CPU
    logging.info(
        f"""Starting SAM...
        Model Type:           {args.model}
        Checkpoint Path:      {os.path.abspath(args.load)}
        Input Path:           {os.path.abspath(args.source)}
        Extension Filter:     {args.ext.split(".")[-1] if args.ext else None}
        Output Path:          {os.path.abspath(args.target)}
        Inference Batch Size: {args.batch_size}
        IO Workers:           {args.num_workers[0]}I{args.num_workers[1]}O
        Annotation Level:     {args.annotation_level}->{"water"+"&sky" if args.annotation_level==2 else ""}
        Output Size:          {args.output_size}
        Multimask:            {args.multimask}
        Inference Device:     {device}
    """
    )

    sam = sam_model_registry[args.model](checkpoint=args.load)
    sam.to(device=device)  # type: ignore
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    logging.info(f"{args.model} loaded.")

    logging.info(f"Scanning {args.source}...")
    ids = []
    for item in tqdm(os.listdir(args.source)):
        id = os.path.join(os.path.abspath(args.source), item)
        if not os.path.isfile(id):  # Non-recursive
            continue
        if args.ext:
            ext = args.ext.split(".")[-1]
            if not id[-len(ext) :] == ext:
                continue
        ids.append(id)
    logging.info(f"Source scan complete, {len(ids)} in total.")

    logging.info(
        f"Loading subprocesses: {args.num_workers[0]}I{args.num_workers[1]}O workers..."
    )
    lQ = Queue(maxsize=max(args.num_workers[0], args.batch_size))  # Loading queue
    wQ = Queue(maxsize=max(args.num_workers[1], args.batch_size))  # Writing queue
    lW = []
    wW = []

    id_slice = len(ids) // args.num_workers[0]
    if len(ids) % args.num_workers[0] > 0:  # A.K.A. math.ceil()
        id_slice += 1

    with tqdm(total=sum(args.num_workers)) as pbar:
        for i in range(args.num_workers[0]):  # Start loading workers
            lW.append(
                Process(
                    target=loading_worker,
                    args=(
                        lQ,
                        ids[id_slice * i : id_slice * (i + 1)],
                        resize_transform,
                        args,
                    ),
                )
            )
            lW[i].start()
            pbar.update(1)
        for i in range(args.num_workers[1]):  # Start writing workers
            wW.append(Process(target=writing_worker, args=(wQ, args)))
            wW[i].start()
            pbar.update(1)
    logging.info(f"Preparation done.")

    batch_count = 0
    bottleneck_count = 0
    bottleneck_threshold = 3  # Max bottleneck_count before warning
    with tqdm(total=len(ids)) as pbar:
        pbar.set_description("Processing")

        done_count = 0
        while done_count < len(ids):
            names = []  # File name

            batched_input = []
            for i in range(args.batch_size):
                if lQ.empty() and batch_count > 0:
                    if bottleneck_count > bottleneck_threshold:
                        logging.warning(
                            f"May consider to increase --num-workers for better GPU utilization."
                        )
                    bottleneck_count += 1
                input_dict = lQ.get(timeout=60)  # 60s loading timeout
                for x in input_dict.keys():
                    if torch.is_tensor(input_dict[x]):
                        input_dict[x] = input_dict[x].to(device)

                names.append(input_dict["name"])
                del input_dict["name"]
                batched_input.append(input_dict)
                done_count += 1

            batched_output = sam(batched_input, multimask_output=args.multimask)

            for i, output_dict in enumerate(batched_output):
                output_dict["name"] = names[i]

                for x in output_dict.keys():
                    if torch.is_tensor(output_dict[x]):
                        output_dict[x] = output_dict[x].to("cpu")
                if wQ.full():
                    if bottleneck_count > bottleneck_threshold:
                        logging.warning(
                            f"May consider to increase --num-workers for better GPU utilization."
                        )
                    bottleneck_count += 1
                wQ.put(output_dict, timeout=60)  # 60s writing timeout

            pbar.update(len(batched_output))
            batch_count += 1

        logging.info(f"{done_count} done! Exiting...")
