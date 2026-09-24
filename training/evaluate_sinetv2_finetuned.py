import os
import sys
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

TEST_IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "Test",
    "Image"
)

TEST_MASK_DIR = os.path.join(
    DATASET_ROOT,
    "Test",
    "GT_Object"
)

SINET_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "sinetv2_finetuned_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINETUNED SINET-V2 COD10K TEST EVALUATION")
print("=" * 70)
print(f"Device: {device}")
print()


# ============================================================
# CHECK DATASET
# ============================================================

print("Checking paths...")

if not os.path.isdir(TEST_IMAGE_DIR):
    raise FileNotFoundError(
        f"Test image folder not found:\n{TEST_IMAGE_DIR}"
    )

if not os.path.isdir(TEST_MASK_DIR):
    raise FileNotFoundError(
        f"Test mask folder not found:\n{TEST_MASK_DIR}"
    )

print("Test Images : OK")
print("Test Masks  : OK")
print()


# ============================================================
# FIND FINETUNED MODEL
# ============================================================

checkpoint_locations = [
    os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "sinetv2_cod10k_best.pth"
    ),
    os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "sinetv2_cod10k_best.pt"
    ),
    os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "sinetv2",
        "sinetv2_cod10k_best.pth"
    ),
    os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "sinetv2",
        "sinetv2_cod10k_best.pt"
    )
]

SINET_CHECKPOINT = None

for path in checkpoint_locations:

    if os.path.isfile(path):
        SINET_CHECKPOINT = path
        break


if SINET_CHECKPOINT is None:

    search_patterns = [
        os.path.join(
            PROJECT_ROOT,
            "**",
            "sinetv2_cod10k_best.pth"
        ),
        os.path.join(
            PROJECT_ROOT,
            "**",
            "sinetv2_cod10k_best.pt"
        )
    ]

    found = []

    for pattern in search_patterns:
        found.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    if found:
        SINET_CHECKPOINT = found[0]


if SINET_CHECKPOINT is None:

    raise FileNotFoundError(
        "Fine-tuned SINet-V2 model could not be found anywhere "
        "inside the Camouflage_Breaker project."
    )


print("Fine-tuned SINet-V2 : OK")
print("Model:")
print(SINET_CHECKPOINT)
print()


# ============================================================
# FIND SINET SOURCE
# ============================================================

network_file = None
res2net_file = None

for root, dirs, files in os.walk(SINET_ROOT):

    if "Network_Res2Net_GRA_NCD.py" in files:
        network_file = os.path.join(
            root,
            "Network_Res2Net_GRA_NCD.py"
        )

    if "Res2Net_v1b.py" in files:
        res2net_file = os.path.join(
            root,
            "Res2Net_v1b.py"
        )


if network_file is None:

    raise FileNotFoundError(
        "SINet-V2 source file Network_Res2Net_GRA_NCD.py "
        "was not found."
    )


if res2net_file is None:

    raise FileNotFoundError(
        "SINet-V2 source file Res2Net_v1b.py "
        "was not found."
    )


SOURCE_LIB = os.path.dirname(
    network_file
)

SOURCE_ROOT = os.path.dirname(
    SOURCE_LIB
)

sys.path.insert(
    0,
    SOURCE_ROOT
)

sys.path.insert(
    0,
    SOURCE_LIB
)

print("SINet-V2 source : OK")
print(SOURCE_ROOT)
print()


# ============================================================
# FIND RES2NET WEIGHTS
# ============================================================

RES2NET_NAME = (
    "res2net50_v1b_26w_4s-3cf99910.pth"
)

res2net_checkpoint = None

for root, dirs, files in os.walk(SINET_ROOT):

    if RES2NET_NAME in files:

        res2net_checkpoint = os.path.join(
            root,
            RES2NET_NAME
        )

        break


if res2net_checkpoint is None:

    raise FileNotFoundError(
        "Res2Net pretrained weights were not found."
    )


print("Res2Net weights : OK")
print(res2net_checkpoint)
print()


# ============================================================
# LOAD RES2NET
# ============================================================

import lib.Res2Net_v1b as res2net_module


original_res2net_function = (
    res2net_module.res2net50_v1b_26w_4s
)


def load_local_res2net(*args, **kwargs):

    kwargs["pretrained"] = False

    model = original_res2net_function(
        *args,
        **kwargs
    )

    checkpoint = torch.load(
        res2net_checkpoint,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]


    cleaned = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned[new_key] = value


    model.load_state_dict(
        cleaned,
        strict=False
    )

    return model


res2net_module.res2net50_v1b_26w_4s = (
    load_local_res2net
)


# ============================================================
# LOAD SINET-V2
# ============================================================

from lib.Network_Res2Net_GRA_NCD import Network


print("Loading fine-tuned SINet-V2...")

model = Network(
    channel=32
)


checkpoint = torch.load(
    SINET_CHECKPOINT,
    map_location=device
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


cleaned_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith("module."):
        new_key = new_key[7:]

    cleaned_state_dict[new_key] = value


model.load_state_dict(
    cleaned_state_dict,
    strict=True
)

model = model.to(device)

model.eval()

print("Fine-tuned SINet-V2 loaded successfully.")
print()


# ============================================================
# PREPROCESSING
# ============================================================

IMAGE_SIZE = 352

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    resized = (
        resized.astype(np.float32)
        / 255.0
    )

    resized = (
        resized - MEAN
    ) / STD

    tensor = torch.from_numpy(
        resized.transpose(2, 0, 1)
    ).float()

    return tensor.unsqueeze(0)


# ============================================================
# METRICS
# ============================================================

def calculate_iou(prediction, ground_truth):

    prediction = prediction > 0
    ground_truth = ground_truth > 0

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(prediction, ground_truth):

    prediction = prediction > 0
    ground_truth = ground_truth > 0

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    total = (
        prediction.sum()
        + ground_truth.sum()
    )

    if total == 0:
        return 1.0

    return (
        2.0 * intersection / total
    )


# ============================================================
# MODEL OUTPUT
# ============================================================

def get_output(output):

    if isinstance(
        output,
        (tuple, list)
    ):
        output = output[-1]

    if output.ndim == 3:
        output = output.unsqueeze(1)

    return output


# ============================================================
# TEST IMAGES
# ============================================================

image_paths = sorted(
    glob.glob(
        os.path.join(
            TEST_IMAGE_DIR,
            "*"
        )
    )
)

image_paths = [
    path
    for path in image_paths
    if path.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )
    )
]


print(
    f"Total test images: {len(image_paths)}"
)

print()
print(
    "Starting full evaluation..."
)
print()


# ============================================================
# EVALUATE
# ============================================================

iou_scores = []
dice_scores = []

missing_masks = 0
failed_images = 0

results = []


with torch.no_grad():

    for image_path in tqdm(
        image_paths,
        desc="Evaluating fine-tuned SINet-V2"
    ):

        filename = os.path.basename(
            image_path
        )

        base_name = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            TEST_MASK_DIR,
            base_name + ".png"
        )

        if not os.path.isfile(
            mask_path
        ):

            mask_path = os.path.join(
                TEST_MASK_DIR,
                base_name + ".jpg"
            )

        if not os.path.isfile(
            mask_path
        ):

            missing_masks += 1
            continue


        image = cv2.imread(
            image_path
        )

        ground_truth = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )


        if (
            image is None
            or ground_truth is None
        ):

            failed_images += 1
            continue


        height, width = image.shape[:2]


        input_tensor = preprocess_image(
            image
        ).to(device)


        output = model(
            input_tensor
        )

        output = get_output(
            output
        )


        prediction = torch.sigmoid(
            output
        )

        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )


        prediction = (
            prediction > 0.5
        ).astype(
            np.uint8
        ) * 255


        prediction = cv2.resize(
            prediction,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )


        ground_truth = (
            ground_truth > 127
        ).astype(
            np.uint8
        ) * 255


        iou = calculate_iou(
            prediction,
            ground_truth
        )

        dice = calculate_dice(
            prediction,
            ground_truth
        )


        iou_scores.append(iou)
        dice_scores.append(dice)


        results.append(
            (
                filename,
                iou,
                dice
            )
        )


# ============================================================
# FINAL RESULTS
# ============================================================

if not iou_scores:

    raise RuntimeError(
        "No test images were successfully evaluated."
    )


mean_iou = float(
    np.mean(iou_scores)
)

mean_dice = float(
    np.mean(dice_scores)
)

minimum_iou = float(
    np.min(iou_scores)
)

maximum_iou = float(
    np.max(iou_scores)
)

iou_50 = int(
    np.sum(
        np.array(iou_scores) >= 0.50
    )
)

iou_70 = int(
    np.sum(
        np.array(iou_scores) >= 0.70
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "sinetv2_finetuned_results.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "image,iou,dice\n"
    )

    for filename, iou, dice in results:

        file.write(
            f"{filename},{iou:.6f},{dice:.6f}\n"
        )


summary_path = os.path.join(
    OUTPUT_DIR,
    "sinetv2_finetuned_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FINETUNED SINET-V2 COD10K TEST RESULTS\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        f"Images evaluated : {len(iou_scores)}\n"
    )

    file.write(
        f"Missing masks    : {missing_masks}\n"
    )

    file.write(
        f"Failed images    : {failed_images}\n"
    )

    file.write(
        f"Mean IoU         : {mean_iou:.4f}\n"
    )

    file.write(
        f"Mean Dice        : {mean_dice:.4f}\n"
    )

    file.write(
        f"Minimum IoU      : {minimum_iou:.4f}\n"
    )

    file.write(
        f"Maximum IoU      : {maximum_iou:.4f}\n"
    )

    file.write(
        f"IoU >= 0.50      : "
        f"{iou_50}/{len(iou_scores)}\n"
    )

    file.write(
        f"IoU >= 0.70      : "
        f"{iou_70}/{len(iou_scores)}\n"
    )


# ============================================================
# PRINT
# ============================================================

print()

print("=" * 70)
print("FINETUNED SINET-V2 FULL RESULTS")
print("=" * 70)

print(
    f"Images evaluated : {len(iou_scores)}"
)

print(
    f"Missing masks    : {missing_masks}"
)

print(
    f"Failed images    : {failed_images}"
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

print()
print("Results saved:")
print(csv_path)
print(summary_path)

print("=" * 70)