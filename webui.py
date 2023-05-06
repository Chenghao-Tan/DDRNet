import os
import subprocess

import blobconverter
import pywebio
import pywebio.input as input
import pywebio.output as output
import pywebio.session as session
import pywebio_battery

from save_onnx import export_onnx

boat_obstacle_avoidance_url = (
    "[Boat-Obstacle-Avoidance](https://github.com/Chenghao-Tan/Boat-Obstacle-Avoidance)"
)
script_dir = os.path.split(os.path.realpath(__file__))[0]


# Generate Dataset
def dataset():
    output.clear()
    output.put_markdown("# Generate Dataset")
    output.put_markdown("`To close the program, press CTRL+C in the console.`")
    output.put_markdown(
        "Use SAM (Segment-Anything-Model) to generate your unique dataset automatically. This step will significantly improve the performance of the model in specific areas."
    )

    input_items = [
        input.select(
            "Model Type",
            name="model",
            help_text="List is in order of increasing accuracy.",
            options=[
                ("vit_b (<4GB VRAM)", "vit_b"),
                ("vit_l (<6GB VRAM)", "vit_l"),
                ("vit_h (<8GB VRAM)", "vit_h"),
            ],
            value="vit_l",
        ),
        input.input(
            "Checkpoint Path",
            name="load",
            help_text="Path to the corresponding model checkpoint (.pth).",
            placeholder="XXX.pth",
            required=True,
            validate=lambda file: "File does not exist!"
            if not (os.path.exists(file) and os.path.isfile(file))
            else (
                "Not a .pth!"
                if os.path.splitext(os.path.split(file)[-1])[-1].lower() != ".pth"
                else None
            ),
        ),
        input.input(
            "Input Directory",
            name="source",
            help_text="Path to the unlabeled image source (reading non-recursively).",
            required=True,
            validate=lambda dir: "Directory is not readable!"
            if not (os.access(dir, os.R_OK) and os.path.isdir(dir))
            else None,
        ),
        input.input(
            "Output Directory",
            name="target",
            help_text="Location to output the dataset (usually is model dataset directory).",
            placeholder=os.path.join(script_dir, "data"),
            value=os.path.join(script_dir, "data"),
            validate=lambda dir: "Directory is not writable!"
            if not (os.access(dir, os.W_OK) and os.path.isdir(dir))
            else None,
        ),
        input.slider(
            "Batch Size",
            name="batch_size",
            help_text="Increase = more VRAM uasage and more speed.",
            value=1,
            min_value=1,
            max_value=64,
        ),
        input.slider(
            "Loading Workers",
            name="num_workers_i",
            help_text="Number of processes loading data.",
            value=1,
            min_value=1,
            max_value=8,
        ),
        input.slider(
            "Writing Workers",
            name="num_workers_o",
            help_text="Number of processes writing data.",
            value=1,
            min_value=1,
            max_value=8,
        ),
        input.radio(
            "Annotation Level",
            name="annotation_level",
            help_text="Select 'Water' if if sky can barely be seen in the images.",
            options=[("Water&Sky", "2"), ("Water", "1")],
            value="2",
            required=True,
        ),
        input.checkbox(
            "No Resize",
            name="no_size",
            help_text="If NOT selected, then the output will be resize to 640x360.",
            options=[("YES", True)],
        ),
        input.checkbox(
            "Visualization Only",
            name="visualize",
            help_text="If selected, then the output cannot be used by training anymore!",
            options=[("YES", True)],
        ),
        input.actions(
            name="operation",
            buttons=[
                {
                    "label": "SUBMIT",
                    "value": "submit",
                    "type": "submit",
                    "color": "primary",
                },
                {"label": "RESET", "type": "reset", "color": "danger"},
                {"label": "HOME", "type": "cancel", "color": "secondary"},
            ],
        ),
    ]
    args = input.input_group("SAM Settings", inputs=input_items)
    if args is None:
        main()  # main() should never block! (in case of stack overflow)
        return

    if pywebio_battery.confirm(
        "Proceed to Generate?",
        "Will use SAM to annotate automatically. It takes a long time.\n",
    ):
        output.put_markdown("## SAM Is Running...")
        pywebio_battery.put_logbox("sam")

        cmd = "python "
        cmd += os.path.join(script_dir, "SAM.py")
        cmd += " --model " + str(args["model"])  # type: ignore
        cmd += " --load " + str(args["load"])  # type: ignore
        cmd += " --source " + str(args["source"])  # type: ignore
        cmd += " --target " + str(args["target"])  # type: ignore
        cmd += " --batch-size " + str(args["batch_size"])  # type: ignore
        cmd += " --num-workers " + str(args["num_workers_i"]) + " " + str(args["num_workers_o"])  # type: ignore
        cmd += " --annotation-level " + str(args["annotation_level"])  # type: ignore
        cmd += " --output-size " + ("0 0" if len(args["no_size"]) else "640 360")  # type: ignore
        cmd += " --visualize" if len(args["visualize"]) else ""  # type: ignore
        pywebio_battery.logbox_append("sam", "> " + cmd)

        # Start subprocess
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if process.stdout:
            for line in iter(process.stdout.readline, b""):
                pywebio_battery.logbox_append("sam", line.decode("utf-8"))
        process.wait()
        output.put_markdown("Generation complete.")
        output.popup("Done!", "Generation complete.")

        # Next step
        output.put_markdown("## Train DDRNet next?")
        output.put_buttons(buttons=["NEXT", "HOME"], onclick=[ddrnet, main])
    else:
        output.put_markdown("## Generation Cancelled")
        output.put_buttons(buttons=["RETRY", "HOME"], onclick=[dataset, main])


# Train DDRNet
def ddrnet():
    output.clear()
    output.put_markdown("# Train DDRNet")
    output.put_markdown("`To close the program, press CTRL+C in the console.`")
    output.put_markdown(
        "Train DDRNet with the dataset under `data` directory. Exporting ONNX with different settings does not require retraining."
    )

    dataset_size = sum(
        [
            os.path.isfile(os.path.join(script_dir, "data", "imgs", x))
            for x in os.listdir(os.path.join(script_dir, "data", "imgs"))
        ]
    )

    def is_number(n: str):
        try:
            float(n)
            return True
        except ValueError:  # Not a number
            pass
        return False

    def check_val(percent):
        nonlocal dataset_size
        if not int(dataset_size * percent / 100) >= 1:
            return "Expect at least one image for validation!"
        elif not int(dataset_size * percent / 100) < dataset_size:
            return "There will be no training data!"
        else:
            return None

    input_items = [
        input.input(
            "Checkpoint Path",
            name="load",
            help_text="Path to the model checkpoint (.pth). Leave it blank if it's a fresh start.",
            placeholder="XXX.pth",
            value="",
            validate=lambda file: None
            if file == ""
            else (
                "File does not exist!"
                if not (os.path.exists(file) and os.path.isfile(file))
                else (
                    "Not a .pth!"
                    if os.path.splitext(os.path.split(file)[-1])[-1].lower() != ".pth"
                    else None
                )
            ),
        ),
        input.slider(
            "Epochs",
            name="epochs",
            help_text="Number of epochs.",
            value=127,
            min_value=1,
            max_value=512,
        ),
        input.slider(
            "Batch Size",
            name="batch_size",
            help_text="Increase = more VRAM uasage and more speed.",
            value=8,
            min_value=1,
            max_value=64,
        ),
        input.input(
            "Learning Rate",
            name="learning_rate",
            help_text="Input learning rate. Support scientific notation.",
            placeholder="3e-4",
            value="3e-4",
            validate=lambda num: "Not a number!"
            if not is_number(num)
            else ("Must be >0!" if not float(num) > 0 else None),
        ),
        input.slider(
            "Validation Proportion",
            name="validation",
            help_text="Percent of the data that is used as validation.",
            value=10,
            min_value=1,
            max_value=99,
            validate=check_val,
        ),
        input.actions(
            name="operation",
            buttons=[
                {
                    "label": "SUBMIT",
                    "value": "submit",
                    "type": "submit",
                    "color": "primary",
                },
                {"label": "RESET", "type": "reset", "color": "danger"},
                {"label": "HOME", "type": "cancel", "color": "secondary"},
            ],
        ),
    ]
    args = input.input_group("Training Settings", inputs=input_items)
    if args is None:
        main()  # main() should never block! (in case of stack overflow)
        return

    valset_num = int(dataset_size * args["validation"] / 100)  # type: ignore
    trainset_num = int(dataset_size - valset_num)
    if pywebio_battery.confirm(
        "Proceed to Train?",
        "Will train DDRNet with {} training images and {} validation images. It will take a long time.\n".format(
            trainset_num, valset_num
        ),
    ):
        output.put_markdown("## Training DDRNet...")
        output.put_markdown(
            "**Please check the logbox for `wandb.ai`'s 'view run' url. Open it and supervise the training process there!**"
        )
        pywebio_battery.put_logbox("train")

        cmd = "python "
        cmd += os.path.join(script_dir, "train.py")
        cmd += (" --load " + str(args["load"])) if args["load"] != "" else ""  # type: ignore
        cmd += " --epochs " + str(args["epochs"])  # type: ignore
        cmd += " --batch-size " + str(args["batch_size"])  # type: ignore
        cmd += " --learning-rate " + str(args["learning_rate"])  # type: ignore
        cmd += " --validation " + str(args["validation"])  # type: ignore
        pywebio_battery.logbox_append("train", "> " + cmd)

        # Start subprocess
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if process.stdout:
            for line in iter(process.stdout.readline, b""):
                pywebio_battery.logbox_append("train", line.decode("utf-8"))
        process.wait()
        output.put_markdown("Training complete.")
        output.put_markdown(
            "The checkpoints are under `checkpoints` directory. Remember to empty that directory afterwards."
        )
        output.popup("Done!", "Training complete.")

        # Next step
        output.put_markdown("## Export ONNX next?")
        output.put_buttons(buttons=["NEXT", "HOME"], onclick=[onnx, main])
    else:
        output.put_markdown("## Training Cancelled")
        output.put_buttons(buttons=["RETRY", "HOME"], onclick=[ddrnet, main])


# Export ONNX
def onnx():
    output.clear()
    output.put_markdown("# Export ONNX")
    output.put_markdown("`To close the program, press CTRL+C in the console.`")
    output.put_markdown(
        "Convert PTH to ONNX using PyTorch's ONNX export feature. (ONNX can be compiled to BLOB in the next step.)"
    )

    input_items = [
        input.input(
            "Input Path",
            name="model",
            help_text="Path to the PTH for input.",
            placeholder="XXX.pth",
            required=True,
            validate=lambda file: "File does not exist!"
            if not (os.path.exists(file) and os.path.isfile(file))
            else (
                "Not a .pth!"
                if os.path.splitext(os.path.split(file)[-1])[-1].lower() != ".pth"
                else None
            ),
        ),
        input.input(
            "Output Directory",
            name="output_dir",
            help_text="Target directory to output the ONNX.",
            required=True,
            validate=lambda dir: "Directory is not writable!"
            if not (os.access(dir, os.W_OK) and os.path.isdir(dir))
            else None,
        ),
        input.input(
            "Resolution Height",
            name="height",
            help_text="Input image(/depth)'s resolution height.",
            placeholder="360",
            value="360",
            validate=lambda num: "Not a number!"
            if not num.isdecimal()
            else ("Must be >0!" if not int(num) > 0 else None),
        ),
        input.input(
            "Resolution Width",
            name="width",
            help_text="Input image(/depth)'s resolution width.",
            placeholder="640",
            value="640",
            validate=lambda num: "Not a number!"
            if not num.isdecimal()
            else ("Must be >0!" if not int(num) > 0 else None),
        ),
        input.checkbox(
            "The Following Are Only Valid When This Is Selected:",
            name="detection",
            help_text="Built-in obstacle detection. Must be compatible with Boat-Obstacle-Avoidance.",
            options=[{"label": "Obstacle Detection", "value": True, "selected": True}],
        ),
        input.slider(
            "Detection - Segmentation Confidence",
            name="confidence",
            help_text="Segmentation confidence threshold.",
            value=0.9,
            min_value=0.0,
            max_value=1.0,
        ),
        input.input(
            "Detection - Grid Num Vertical",
            name="grid_num_h",
            help_text="Grid number on the vertical axis, must be divisible by 'Resolution Height'. Must be compatible with Boat-Obstacle-Avoidance.",
            placeholder="5",
            value="5",
            validate=lambda num: "Not a number!"
            if not num.isdecimal()
            else ("Must be >0!" if not int(num) > 0 else None),
        ),
        input.input(
            "Detection - Grid Num Horizontal",
            name="grid_num_w",
            help_text="Grid number on the horizontal axis, must be divisible by 'Resolution Width'. Must be compatible with Boat-Obstacle-Avoidance.",
            placeholder="5",
            value="5",
            validate=lambda num: "Not a number!"
            if not num.isdecimal()
            else ("Must be >0!" if not int(num) > 0 else None),
        ),
        input.slider(
            "Detection - Pixel Threshold",
            name="threshold",
            help_text="When the proportion of obstacle pixels in the grid reaches this threshold, the grid is considered an obstacle.",
            value=0.2,
            min_value=0.0,
            max_value=1.0,
        ),
        input.actions(
            name="operation",
            buttons=[
                {
                    "label": "SUBMIT",
                    "value": "submit",
                    "type": "submit",
                    "color": "primary",
                },
                {"label": "RESET", "type": "reset", "color": "danger"},
                {
                    "label": "HOME",
                    "type": "cancel",
                    "color": "secondary",
                },
            ],
        ),
    ]
    args = input.input_group("Export Settings", inputs=input_items)
    if args is None:
        main()  # main() should never block! (in case of stack overflow)
        return

    output.put_markdown("## Exporting...")
    with output.put_loading(color="primary"):
        onnx_name = os.path.splitext(os.path.split(args["model"])[-1])[0] + ".onnx"  # type: ignore
        onnx_path = os.path.join(args["output_dir"], onnx_name)  # type: ignore
        export_onnx(
            model=args["model"],  # type: ignore
            output=onnx_path,
            height=int(args["height"]),  # type: ignore
            width=int(args["width"]),  # type: ignore
            detection=True if len(args["detection"]) else False,  # type: ignore
            confidence=args["confidence"],  # type: ignore
            grid_height=int(args["height"]) // int(args["grid_num_h"]),  # type: ignore
            grid_width=int(args["width"]) // int(args["grid_num_w"]),  # type: ignore
            grid_threshold=args["threshold"],  # type: ignore
        )
    output.put_markdown(f"Successfully exported. Target ONNX: {onnx_path}")

    # Next step
    output.put_markdown("## Convert to BLOB next?")
    output.put_buttons(buttons=["NEXT", "HOME"], onclick=[blob, main])


# Convert to BLOB
def blob():
    output.clear()
    output.put_markdown("# Convert to BLOB")
    output.put_markdown("`To close the program, press CTRL+C in the console.`")
    output.put_markdown(
        f"Convert ONNX to BLOB using [Luxonis's official online tool](https://blobconverter.luxonis.com/). The BLOB is for {boat_obstacle_avoidance_url}."
    )

    input_items = [
        input.input(
            "Input Path",
            name="model",
            help_text="Path to the ONNX for input.",
            placeholder="XXX.onnx",
            required=True,
            validate=lambda file: "File does not exist!"
            if not (os.path.exists(file) and os.path.isfile(file))
            else (
                "Not a .onnx!"
                if os.path.splitext(os.path.split(file)[-1])[-1].lower() != ".onnx"
                else None
            ),
        ),
        input.input(
            "Output Directory",
            name="output_dir",
            help_text="Target directory to output the BLOB.",
            required=True,
            validate=lambda dir: "Directory is not writable!"
            if not (os.access(dir, os.W_OK) and os.path.isdir(dir))
            else None,
        ),
        input.slider(
            "Shaves",
            name="shaves",
            help_text="How many SHAVE cores to use. (CMX_SLICES is also this many.)",
            value=6,
            min_value=1,
            max_value=16,
        ),
        input.select(
            "OpenVINO Version",
            name="version",
            help_text="Select which version of OpenVINO to use. (Don't change this if you don't know what you are doing!)",
            options=["2021.4", "2022.1"],
            value="2021.4",
        ),
        input.input(
            "Model Optimizer Params",
            name="optimizer_params",
            help_text="Written in the format of cmd args. (Don't change this if you don't know what you are doing!)",
            placeholder="--data_type=FP16 has always been set by default.",
            value="",
        ),
        input.input(
            "MyriadX Compile Params",
            name="compile_params",
            help_text="Written in the format of cmd args. (Don't change this if you don't know what you are doing!)",
            place_holder="-ip U8",
            value="-ip U8",
        ),
        input.actions(
            name="operation",
            buttons=[
                {
                    "label": "SUBMIT",
                    "value": "submit",
                    "type": "submit",
                    "color": "primary",
                },
                {"label": "RESET", "type": "reset", "color": "danger"},
                {"label": "HOME", "type": "cancel", "color": "secondary"},
            ],
        ),
    ]
    args = input.input_group("Conversion Settings", inputs=input_items)
    if args is None:
        main()  # main() should never block! (in case of stack overflow)
        return

    output.put_markdown("## Converting...")
    pywebio_battery.put_logbox("blobconverter")

    blob_config = {
        "model": args["model"],  # type: ignore
        "version": args["version"],  # type: ignore
        "shaves": args["shaves"],  # type: ignore
        "output_dir": args["output_dir"],  # type: ignore
        "compile_params": [args["compile_params"]],  # type: ignore
        "data_type": "FP16",
        "optimizer_params": [] if args["optimizer_params"] == "" else [args["optimizer_params"]],  # type: ignore
        "use_cache": False,
    }

    def logbox_print(msg, name="blobconverter"):
        try:
            pywebio_battery.logbox_append(name, msg + "\n")
        except:
            pass  # Because stdout is redirected, nothing will be printed when exiting

    with pywebio_battery.redirect_stdout(output_func=logbox_print):  # type: ignore
        blob_name = blobconverter.from_onnx(**blob_config)

    blob_path = os.path.join(args["output_dir"], blob_name)  # type: ignore
    if blobconverter.is_valid_blob(blob_path):
        output.put_markdown(f"Successfully converted. Target BLOB: {blob_path}")

        # Next step
        output.put_markdown("## Seems that it's all done!")
        output.put_buttons(buttons=["HOME"], onclick=[main])
    else:
        output.put_markdown("Conversion failed.")
        output.put_buttons(buttons=["RETRY", "HOME"], onclick=[blob, main])


# Main
def main():
    output.clear()
    output.put_markdown("# Model Workbench")
    output.put_markdown("`To close the program, press CTRL+C in the console.`")
    output.put_markdown(
        f"Create the AI model required by {boat_obstacle_avoidance_url}. From left to right is the complete process in normal. ('Generate Dataset' is optional.)"
    )

    output.put_markdown("## Select to start:")
    output.put_buttons(
        buttons=[
            "Generate Dataset",
            "Train DDRNet",
            "Export ONNX",
            "Convert to BLOB",
        ],
        onclick=[dataset, ddrnet, onnx, blob],
    )


# Do some settings before open
def with_settings(open=main):
    # Page title
    session.set_env(title="Model Workbench", auto_scroll_bottom=True)
    # Auto dark mode
    session.run_js(
        """
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.style.backgroundColor = '#1c1c1c';
            document.body.style.color = '#eeeeee';
            const footer = document.querySelector("footer");
            footer.style.backgroundColor = '#1c1c1c';
        }
    """
    )
    # Then open
    open()


if __name__ == "__main__":
    try:
        pywebio.start_server(with_settings, port=0, auto_open_webbrowser=True)
    except KeyboardInterrupt:
        print("Exiting...")
