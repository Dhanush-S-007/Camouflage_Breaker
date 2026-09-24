import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm


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

SINET_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth"
)

RES2NET_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "res2net50_v1b_26w_4s-3cf99910.pth"
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
    "sinetv2_evaluation"
)

IMAGE_SIZE = 352

MAX_IMAGES = 20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


sys.path.insert(
    0,
    SINET_ROOT
)


def load_sinetv2():

    import lib.Res2Net_v1b as res2net_module

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

            model_state = torch.load(
                RES2NET_CHECKPOINT,
                map_location="cpu"
            )

            if (
                isinstance(model_state, dict)
                and "state_dict" in model_state
            ):

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

    from lib.Network_Res2Net_GRA_NCD import Network

    model = Network(
        channel=32
    )

    checkpoint = torch.load(
        SINET_CHECKPOINT,
        map_location="cpu"
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        checkpoint = checkpoint[
            "state_dict"
        ]

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

    return model


def calculate_iou(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        bool
    )

    ground_truth = ground_truth.astype(
        bool
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:

        return 0.0

    return intersection / union


def calculate_dice(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        bool
    )

    ground_truth = ground_truth.astype(
        bool
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    total = (
        prediction.sum()
        + ground_truth.sum()
    )

    if total == 0:

        return 0.0

    return (
        2 * intersection
    ) / total


def find_mask(
    image_name
):

    image_base = os.path.splitext(
        image_name
    )[0]

    for mask_name in os.listdir(
        MASK_DIR
    ):

        mask_base = os.path.splitext(
            mask_name
        )[0]

        if mask_base == image_base:

            return os.path.join(
                MASK_DIR,
                mask_name
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

    print("=" * 70)
    print("SINet-V2 COD10K EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()

    if not os.path.exists(
        SINET_CHECKPOINT
    ):

        print(
            "ERROR: SINet-V2 checkpoint not found:"
        )

        print(
            SINET_CHECKPOINT
        )

        return

    if not os.path.exists(
        RES2NET_CHECKPOINT
    ):

        print(
            "ERROR: Res2Net checkpoint not found:"
        )

        print(
            RES2NET_CHECKPOINT
        )

        return

    print(
        "Loading SINet-V2..."
    )

    model = load_sinetv2()

    model = model.to(
        device
    )

    model.eval()

    print(
        "SINet-V2 loaded."
    )

    print()

    image_files = sorted([
        file_name
        for file_name in os.listdir(
            IMAGE_DIR
        )
        if file_name.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ])

    image_files = image_files[
        :MAX_IMAGES
    ]

    print(
        f"Images selected: "
        f"{len(image_files)}"
    )

    print()

    iou_scores = []
    dice_scores = []

    results = []

    for image_name in tqdm(
        image_files,
        desc="Evaluating SINet-V2"
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        mask_path = find_mask(
            image_name
        )

        if mask_path is None:

            continue

        image = cv2.imread(
            image_path
        )

        ground_truth = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            continue

        if ground_truth is None:

            continue

        original_height, original_width = (
            image.shape[:2]
        )

        input_tensor = preprocess_image(
            image
        ).to(device)

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

        binary_prediction = (
            prediction > 0.5
        ).astype(
            np.uint8
        )

        binary_ground_truth = (
            ground_truth > 127
        ).astype(
            np.uint8
        )

        iou = calculate_iou(
            binary_prediction,
            binary_ground_truth
        )

        dice = calculate_dice(
            binary_prediction,
            binary_ground_truth
        )

        iou_scores.append(
            iou
        )

        dice_scores.append(
            dice
        )

        results.append({
            "image": image_name,
            "iou": iou,
            "dice": dice
        })

    if not iou_scores:

        print()
        print(
            "ERROR: No images were evaluated."
        )

        return

    mean_iou = np.mean(
        iou_scores
    )

    mean_dice = np.mean(
        dice_scores
    )

    minimum_iou = np.min(
        iou_scores
    )

    maximum_iou = np.max(
        iou_scores
    )

    iou_50 = np.sum(
        np.array(iou_scores) >= 0.50
    )

    iou_70 = np.sum(
        np.array(iou_scores) >= 0.70
    )

    print()

    print("=" * 70)
    print("SINet-V2 RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated : {len(iou_scores)}"
    )

    print(
        f"Mean IoU         : {mean_iou:.4f}"
    )

    print(
        f"Mean Dice        : {mean_dice:.4f}"
    )

    print(
        f"Minimum IoU      : {minimum_iou:.4f}"
    )

    print(
        f"Maximum IoU      : {maximum_iou:.4f}"
    )

    print(
        f"IoU >= 0.50      : "
        f"{iou_50}/{len(iou_scores)}"
    )

    print(
        f"IoU >= 0.70      : "
        f"{iou_70}/{len(iou_scores)}"
    )

    print("=" * 70)

    results_path = os.path.join(
        OUTPUT_DIR,
        "sinetv2_results.csv"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "image,iou,dice\n"
        )

        for result in results:

            file.write(
                f"{result['image']},"
                f"{result['iou']:.6f},"
                f"{result['dice']:.6f}\n"
            )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "sinetv2_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SINet-V2 COD10K Evaluation\n"
        )

        file.write(
            f"Images evaluated: "
            f"{len(iou_scores)}\n"
        )

        file.write(
            f"Mean IoU: "
            f"{mean_iou:.6f}\n"
        )

        file.write(
            f"Mean Dice: "
            f"{mean_dice:.6f}\n"
        )

        file.write(
            f"Minimum IoU: "
            f"{minimum_iou:.6f}\n"
        )

        file.write(
            f"Maximum IoU: "
            f"{maximum_iou:.6f}\n"
        )

        file.write(
            f"IoU >= 0.50: "
            f"{iou_50}/{len(iou_scores)}\n"
        )

        file.write(
            f"IoU >= 0.70: "
            f"{iou_70}/{len(iou_scores)}\n"
        )

    print()

    print(
        "Saved:"
    )

    print(
        results_path
    )

    print(
        summary_path
    )

    print("=" * 70)


if __name__ == "__main__":
    main()