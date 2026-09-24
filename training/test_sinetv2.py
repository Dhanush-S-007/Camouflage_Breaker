import os
import sys
import cv2
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SINET_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "Image"
)

MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "GT_Object"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "sinetv2_test"
)

IMAGE_SIZE = 352

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def find_res2net_checkpoint():

    target_name = (
        "res2net50_v1b_26w_4s-3cf99910.pth"
    )

    search_roots = [
        os.path.join(
            PROJECT_ROOT,
            "models",
            "sinetv2"
        ),
        PROJECT_ROOT
    ]

    checked = set()

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):

            for file_name in files:

                if file_name.lower() == target_name.lower():

                    full_path = os.path.join(
                        current_root,
                        file_name
                    )

                    full_path = os.path.abspath(
                        full_path
                    )

                    if full_path not in checked:

                        return full_path

                    checked.add(full_path)

    return None


RES2NET_CHECKPOINT = find_res2net_checkpoint()


def patch_res2net_loader():

    if RES2NET_CHECKPOINT is None:

        raise FileNotFoundError(
            "Res2Net checkpoint was not found anywhere inside "
            "C:\\Camouflage_Breaker\\models\\sinetv2"
        )

    sys.path.insert(
        0,
        SINET_ROOT
    )

    import lib.Res2Net_v1b as res2net_module

    original_function = (
        res2net_module.res2net50_v1b_26w_4s
    )

    def local_res2net50_v1b_26w_4s(
        pretrained=False,
        **kwargs
    ):

        model = res2net_module.Res2Net(
            res2net_module.Bottle2neck,
            [3, 4, 6, 3],
            baseWidth=26,
            scale=4,
            **kwargs
        )

        if pretrained:

            print(
                "Loading Res2Net backbone:"
            )

            print(
                RES2NET_CHECKPOINT
            )

            model_state = torch.load(
                RES2NET_CHECKPOINT,
                map_location="cpu"
            )

            if isinstance(
                model_state,
                dict
            ) and "state_dict" in model_state:

                model_state = (
                    model_state["state_dict"]
                )

            cleaned_state = {}

            for key, value in model_state.items():

                new_key = key

                if new_key.startswith(
                    "module."
                ):

                    new_key = new_key[
                        len("module.") :
                    ]

                cleaned_state[
                    new_key
                ] = value

            model.load_state_dict(
                cleaned_state,
                strict=True
            )

        return model

    res2net_module.res2net50_v1b_26w_4s = (
        local_res2net50_v1b_26w_4s
    )

    return original_function


patch_res2net_loader()

from lib.Network_Res2Net_GRA_NCD import Network


def calculate_iou(
    pred,
    gt
):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    if union == 0:

        return 0.0

    return intersection / union


def calculate_dice(
    pred,
    gt
):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    total = (
        pred.sum()
        + gt.sum()
    )

    if total == 0:

        return 0.0

    return (
        2 * intersection
    ) / total


def find_ground_truth(
    image_name
):

    base_name = os.path.splitext(
        image_name
    )[0]

    for file_name in os.listdir(
        MASK_DIR
    ):

        file_base = os.path.splitext(
            file_name
        )[0]

        if file_base == base_name:

            return os.path.join(
                MASK_DIR,
                file_name
            )

    return None


def preprocess_image(
    image
):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_resized = (
        image_resized.astype(
            np.float32
        ) / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    image_resized = (
        image_resized - mean
    ) / std

    tensor = torch.from_numpy(
        image_resized
    ).permute(
        2,
        0,
        1
    ).unsqueeze(0)

    return tensor


def main():

    print("=" * 60)
    print("SINet-V2 CAMOUFLAGE TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()

    if RES2NET_CHECKPOINT is None:

        print(
            "ERROR: Res2Net checkpoint not found."
        )

        print(
            "Expected filename:"
        )

        print(
            "res2net50_v1b_26w_4s-3cf99910.pth"
        )

        return

    print(
        "Res2Net checkpoint:"
    )

    print(
        RES2NET_CHECKPOINT
    )

    print()

    if not os.path.exists(
        CHECKPOINT
    ):

        print(
            "ERROR: SINet-V2 checkpoint not found:"
        )

        print(
            CHECKPOINT
        )

        return

    image_files = sorted([
        f
        for f in os.listdir(
            IMAGE_DIR
        )
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ])

    if not image_files:

        print(
            "ERROR: No test images found."
        )

        return

    image_name = image_files[0]

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    gt_path = find_ground_truth(
        image_name
    )

    if gt_path is None:

        print(
            "ERROR: Ground-truth mask not found."
        )

        return

    print(
        f"Image: {image_name}"
    )

    print(
        f"Ground Truth: "
        f"{os.path.basename(gt_path)}"
    )

    print()

    print(
        "Loading SINet-V2..."
    )

    model = Network(
        channel=32
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        checkpoint = (
            checkpoint["state_dict"]
        )

    cleaned_checkpoint = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_checkpoint[
            new_key
        ] = value

    model.load_state_dict(
        cleaned_checkpoint,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "SINet-V2 loaded."
    )

    image = cv2.imread(
        image_path
    )

    gt = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    if gt is None:

        print(
            "ERROR: Could not read ground-truth."
        )

        return

    original_height, original_width = (
        image.shape[:2]
    )

    input_tensor = preprocess_image(
        image
    ).to(device)

    print(
        "Running SINet-V2..."
    )

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

        if isinstance(
            prediction,
            (tuple, list)
        ):

            prediction = prediction[-1]

        prediction = torch.sigmoid(
            prediction
        )

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    prediction = cv2.resize(
        prediction,
        (
            original_width,
            original_height
        )
    )

    binary_mask = (
        prediction > 0.5
    ).astype(
        np.uint8
    )

    gt_binary = (
        gt > 127
    ).astype(
        np.uint8
    )

    iou = calculate_iou(
        binary_mask,
        gt_binary
    )

    dice = calculate_dice(
        binary_mask,
        gt_binary
    )

    print()

    print("=" * 60)
    print("SINet-V2 RESULTS")
    print("=" * 60)

    print(
        f"IoU : {iou:.4f}"
    )

    print(
        f"Dice: {dice:.4f}"
    )

    print("=" * 60)

    mask_path = os.path.join(
        OUTPUT_DIR,
        "sinetv2_mask.png"
    )

    cv2.imwrite(
        mask_path,
        binary_mask * 255
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    overlay = image_rgb.copy()

    overlay[binary_mask == 1] = (
        overlay[binary_mask == 1] * 0.5
        + np.array(
            [0, 255, 0]
        ) * 0.5
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        "sinetv2_overlay.jpg"
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        mask_path
    )

    print(
        overlay_path
    )

    print("=" * 60)


if __name__ == "__main__":
    main()