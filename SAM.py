import argparse
import logging
import os
from multiprocessing import Process, Queue, Value

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
        help="Size of the output images and masks (WxH) (0 0 = unchanged)",
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the output mask (for debug)",
    )
    return parser.parse_args()


def loading_worker(lQ, index, ids, transform, args):
    try:
        while True:
            with index.get_lock():
                if index.value >= len(ids):  # If all done
                    break
                else:
                    id = ids[index.value]  # Read current id
                    index.value += 1  # Set index to the next one

            name = os.path.basename(id)
            if not os.path.exists(id):
                logging.error(f"{id} does not exist!")
                continue  # Just skip it
            image_raw = Image.open(id)
            image_size = (image_raw.size[1], image_raw.size[0])  # H x W

            image = np.array(image_raw)
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
                    "raw": image_raw,
                },
            )
    except:
        logging.exception(f"An error occured in a loading worker:")
    finally:
        lQ.put({})  # Inform the main process of its exiting


def visualize(image, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    colors = {
        0: torch.tensor([255, 0, 0, 128], dtype=torch.uint8, device=device),  # Red
        1: torch.tensor([0, 255, 0, 128], dtype=torch.uint8, device=device),  # Green
        2: torch.tensor([0, 0, 255, 128], dtype=torch.uint8, device=device),  # Blue
    }

    mask = torch.tensor(np.array(mask), dtype=torch.uint8, device=device)
    top = torch.zeros((*mask.shape, 4), dtype=torch.uint8, device=device)

    for value, color in colors.items():
        condition = (mask == value).unsqueeze(-1)
        color_expanded = color.unsqueeze(0).unsqueeze(1).expand_as(top)
        top = torch.where(condition, color_expanded, top)

    top = Image.fromarray(top.cpu().numpy(), mode="RGBA")
    return Image.alpha_composite(image.convert("RGBA"), top).convert("RGB")


def writing_worker(wQ, args):
    try:
        while True:
            output_dict = wQ.get()
            if len(output_dict) == 0:  # Signal for exiting
                break  # Exit

            name = output_dict["name"]
            image_raw = output_dict["raw"]
            mask = output_dict["mask"]

            mask = tf.ToPILImage()(mask)
            if args.output_size[0] and args.output_size[1]:
                image_raw = image_raw.resize(args.output_size)  # WxH
                mask = mask.resize(args.output_size, resample=Image.NEAREST)  # WxH

            imgs_path = os.path.join(os.path.abspath(args.target), "imgs")
            masks_path = os.path.join(os.path.abspath(args.target), "masks")
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)  # Make sure the writing path is valid
                logging.warning(f"Output directory created: {imgs_path}.")
            if not os.path.exists(masks_path):
                os.makedirs(masks_path)  # Make sure the writing path is valid
                logging.warning(f"Output directory created: {masks_path}.")
            if not args.visualize:  # No image_raw in debug output
                image_raw.save(
                    os.path.join(imgs_path, f"{name}.png"),
                    quality=100,  # In case of jpg
                )  # Save image
            if args.visualize:  # Debug output
                mask = visualize(image_raw, mask)
            mask.save(
                os.path.join(masks_path, f"{name}.png"), quality=100  # In case of jpg
            )  # Save mask
    except:
        logging.exception(f"An error occured in a writing worker:")


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(
        f"""Starting SAM...
        Model Type:           {args.model}
        Checkpoint Path:      {os.path.abspath(args.load)}
        Input Path:           {os.path.abspath(args.source)}
        Extension Filter:     {args.ext.split(".")[-1] if args.ext else None}
        Output Path:          {os.path.abspath(args.target)}
        Inference Batch Size: {args.batch_size}
        IO Workers:           {args.num_workers[0]}I{args.num_workers[1]}O
        Annotation Level:     {args.annotation_level}->{"water"+("&sky" if args.annotation_level==2 else "")}
        Output Size:          {args.output_size if args.output_size[0] and args.output_size[1] else "Unchanged"}
        Multimask:            {args.multimask}
        Visualize:            {args.visualize}
        Inference Device:     {device}
    """
    )

    sam = sam_model_registry[args.model](checkpoint=args.load)
    sam.to(device=device)  # type: ignore
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    logging.info(f"{args.model} loaded.")

    logging.info(f"Scanning {args.source}...")
    ids = []
    for item in tqdm(os.listdir(args.source), disable=None):
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

    loading_index = Value("Q", 0, lock=True)  # Current loading index

    with tqdm(total=sum(args.num_workers), disable=None) as pbar:
        for i in range(args.num_workers[0]):  # Start loading workers
            worker = Process(
                target=loading_worker,
                args=(lQ, loading_index, ids, resize_transform, args),
                daemon=True,
            )
            worker.start()
            lW.append(worker)
            pbar.update(1)
        for i in range(args.num_workers[1]):  # Start writing workers
            worker = Process(target=writing_worker, args=(wQ, args))
            worker.start()
            wW.append(worker)
            pbar.update(1)
    logging.info(f"Preparation done.")

    try:
        bottleneck_count_I = bottleneck_count_O = 0
        bottleneck_threshold = 3  # Max bottleneck_count before warning
        with tqdm(total=len(ids), disable=None) as pbar:
            pbar.set_description("Processing")

            done_count = 0  # Number of the saved
            bad_count = 0  # Number of the loaded but not saved
            exit = 0  # Loading worker exit counter
            while True:
                raw = []  # [(file_name, raw_PIL_Image), ...]

                # Load a batch
                batched_input = []
                while True:
                    # Get input data
                    if lQ.empty() and done_count > 0:
                        if bottleneck_count_I > bottleneck_threshold:
                            logging.warning(
                                f"May consider to increase --num-workers I _ for better GPU utilization."
                            )
                        bottleneck_count_I += 1
                    input_dict = lQ.get(timeout=60)  # 60s loading timeout

                    if len(input_dict) == 0:  # If it is an empty dict
                        exit += 1  # Then it means a loading worker exited
                        if exit == args.num_workers[0]:  # If no more data is coming
                            break
                    else:
                        # Move to GPU
                        for x in input_dict.keys():
                            if torch.is_tensor(input_dict[x]):
                                input_dict[x] = input_dict[x].to(device)

                        # Pre-process
                        # Pass through necessary things
                        raw.append((input_dict["name"], input_dict["raw"]))
                        del input_dict["name"]
                        del input_dict["raw"]

                        batched_input.append(input_dict)
                        if len(batched_input) == args.batch_size:  # If a batch is ready
                            break

                saved_count = len(batched_input)
                jumped_count = 0
                if len(batched_input):  # If input is not empty
                    # Inference
                    batched_output = sam(batched_input, multimask_output=args.multimask)

                    # Unpack a batch
                    for i, output_dict in enumerate(batched_output):
                        # Post-process
                        # Select the best masks (for multimask feature)
                        masks = output_dict["masks"]
                        scores = output_dict["iou_predictions"]
                        if masks.shape[1] > 1:  # A.K.A. args.multimask==True
                            masks = masks[
                                torch.arange(masks.shape[0]), scores.argmax(dim=1)
                            ]  # Select masks with the highest scores
                        else:
                            masks = masks[:, 0, :, :]  # A.K.A. masks.squeeze(dim=1)

                        # Post-process
                        # Merge multi-classes into one mask per image
                        mask = masks[0].type(torch.uint8)  # 1->water
                        for j, m in enumerate(masks[1:], start=2):
                            mask += m * j  # 2->sky, 3->more...
                        mask[mask > len(masks)] = 4  # Overlap defaults to 4 (unknown)

                        # Post-process
                        # Sometimes the prompt point(s) happen to be on the obstacles instead of water/sky/...
                        # It's also why you should set annotation level carefully.
                        threshold = 0.25  # Proportion of water&sky&... should be greater than threshold
                        if ((mask > 0) & (mask != 4)).sum() / (
                            mask.shape[0] * mask.shape[1]
                        ) < threshold:
                            saved_count -= 1
                            jumped_count += 1
                            logging.warning(
                                f"Jumped one image ({raw[i][0]}) due to poor annotation quality."
                            )
                            continue

                        # Post-process
                        # Rebuild output dict
                        output_dict = {
                            "name": raw[i][0],
                            "raw": raw[i][1],
                            "mask": mask,
                        }

                        # Move to CPU
                        for x in output_dict.keys():
                            if torch.is_tensor(output_dict[x]):
                                output_dict[x] = output_dict[x].to("cpu")

                        # Send output data
                        if wQ.full():
                            if bottleneck_count_O > bottleneck_threshold:
                                logging.warning(
                                    f"May consider to increase --num-workers _ O for better GPU utilization."
                                )
                            bottleneck_count_O += 1
                        wQ.put(output_dict, timeout=60)  # 60s writing timeout

                pbar.update(saved_count + jumped_count)
                done_count += saved_count
                bad_count += jumped_count
                if pbar.disable:
                    logging.info(
                        f"Processing: {round((done_count+bad_count)/len(ids)*100,1)}%"
                    )
                if exit == args.num_workers[0]:  # If no more data is coming
                    pbar.update(
                        len(ids) - (done_count + bad_count)
                    )  # Forced set progress to 100%
                    if pbar.disable:
                        logging.info(f"Processing: 100.0%")
                    break

        logging.info(f"{done_count} done! Exiting...")
    except:
        logging.exception(f"An error occured:")
    finally:
        try:
            for i in wW:  # Try to let writing workers finish
                wQ.put({}, timeout=60)  # 60s writing timeout
        except:
            for i in wW:  # Terminate writing workers
                if i.is_alive():
                    i.terminate()
        finally:
            for i in wW:  # Wait for writing workers to exit
                i.join()
