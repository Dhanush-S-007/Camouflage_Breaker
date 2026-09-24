import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SINET_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

RES2NET_CHECKPOINT = None

for root, dirs, files in os.walk(
    os.path.join(PROJECT_ROOT, "models", "sinetv2")
):
    if "res2net50_v1b_26w_4s-3cf99910.pth" in files:
        RES2NET_CHECKPOINT = os.path.join(
            root,
            "res2net50_v1b_26w_4s-3cf99910.pth"
        )
        break

if RES2NET_CHECKPOINT is None:
    raise FileNotFoundError(
        "Res2Net checkpoint not found under models/sinetv2"
    )

sys.path.insert(0, SINET_ROOT)

import lib.Res2Net_v1b as res2net_module


_original_res2net = res2net_module.res2net50_v1b_26w_4s


def local_res2net50_v1b_26w_4s(pretrained=False):
    model = _original_res2net(pretrained=False)

    checkpoint = torch.load(
        RES2NET_CHECKPOINT,
        map_location="cpu"
    )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned_checkpoint = {}

    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned_checkpoint[key] = value

    model.load_state_dict(
        cleaned_checkpoint,
        strict=False
    )

    return model


res2net_module.res2net50_v1b_26w_4s = local_res2net50_v1b_26w_4s

from lib.Network_Res2Net_GRA_NCD import Network


TEST_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "Image"
)

TEST_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "GT_Object"
)

SINET_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "sinetv2_full_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total


print("=" * 70)
print("SINET-V2 FULL COD10K TEST EVALUATION")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

print("\nChecking paths...")

if not os.path.isdir(TEST_IMAGE_DIR):
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_IMAGE_DIR}"
    )

if not os.path.isdir(TEST_MASK_DIR):
    raise FileNotFoundError(
        f"Test mask directory not found:\n{TEST_MASK_DIR}"
    )

if not os.path.isfile(SINET_CHECKPOINT):
    raise FileNotFoundError(
        f"SINet-V2 checkpoint not found:\n{SINET_CHECKPOINT}"
    )

print("Test Images: OK")
print("Test Masks : OK")
print("SINet-V2   : OK")

image_files = sorted([
    f for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

mask_lookup = {}

for filename in os.listdir(TEST_MASK_DIR):
    base_name = os.path.splitext(filename)[0]
    mask_lookup[base_name] = os.path.join(
        TEST_MASK_DIR,
        filename
    )

print(f"\nTotal test images: {len(image_files)}")

print("\nLoading SINet-V2...")

model = Network(
    channel=32
)

checkpoint = torch.load(
    SINET_CHECKPOINT,
    map_location=device
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

cleaned_checkpoint = {}

for key, value in checkpoint.items():
    if key.startswith("module."):
        key = key[7:]

    cleaned_checkpoint[key] = value

model.load_state_dict(
    cleaned_checkpoint,
    strict=False
)

model.to(device)
model.eval()

print("SINet-V2 loaded successfully.")

mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

results = []

missing_masks = 0

print("\nStarting full evaluation...\n")

with torch.no_grad():

    for filename in tqdm(
        image_files,
        desc="Evaluating SINet-V2"
    ):

        image_path = os.path.join(
            TEST_IMAGE_DIR,
            filename
        )

        base_name = os.path.splitext(filename)[0]

        if base_name not in mask_lookup:
            missing_masks += 1
            continue

        mask_path = mask_lookup[base_name]

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_height, original_width = image.shape[:2]

        resized = cv2.resize(
            image,
            (352, 352),
            interpolation=cv2.INTER_LINEAR
        )

        resized = resized.astype(np.float32) / 255.0

        resized = (
            resized - mean
        ) / std

        tensor = torch.from_numpy(
            resized.transpose(2, 0, 1)
        ).float()

        tensor = tensor.unsqueeze(0).to(device)

        output = model(tensor)

        if isinstance(output, (list, tuple)):
            output = output[-1]

        prediction = torch.sigmoid(output)

        prediction = prediction.squeeze().cpu().numpy()

        prediction = (
            prediction > 0.5
        ).astype(np.uint8) * 255

        prediction = cv2.resize(
            prediction,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        gt_mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if gt_mask is None:
            continue

        if gt_mask.shape != prediction.shape:
            gt_mask = cv2.resize(
                gt_mask,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )

        iou = calculate_iou(
            prediction,
            gt_mask
        )

        dice = calculate_dice(
            prediction,
            gt_mask
        )

        results.append({
            "image": filename,
            "iou": iou,
            "dice": dice
        })


results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "sinetv2_full_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

mean_iou = results_df["iou"].mean()
mean_dice = results_df["dice"].mean()

min_iou = results_df["iou"].min()
max_iou = results_df["iou"].max()

iou_50 = (
    results_df["iou"] >= 0.50
).sum()

iou_70 = (
    results_df["iou"] >= 0.70
).sum()

summary_path = os.path.join(
    OUTPUT_DIR,
    "sinetv2_full_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SINet-V2 FULL COD10K TEST EVALUATION\n"
    )
    f.write("=" * 60 + "\n")
    f.write(
        f"Images evaluated : {len(results_df)}\n"
    )
    f.write(
        f"Missing masks    : {missing_masks}\n"
    )
    f.write(
        f"Mean IoU         : {mean_iou:.4f}\n"
    )
    f.write(
        f"Mean Dice        : {mean_dice:.4f}\n"
    )
    f.write(
        f"Minimum IoU      : {min_iou:.4f}\n"
    )
    f.write(
        f"Maximum IoU      : {max_iou:.4f}\n"
    )
    f.write(
        f"IoU >= 0.50      : {iou_50}/{len(results_df)}\n"
    )
    f.write(
        f"IoU >= 0.70      : {iou_70}/{len(results_df)}\n"
    )


print("\n")
print("=" * 70)
print("SINET-V2 FULL RESULTS")
print("=" * 70)

print(
    f"Images evaluated : {len(results_df)}"
)

print(
    f"Missing masks    : {missing_masks}"
)

print(
    f"Mean IoU         : {mean_iou:.4f}"
)

print(
    f"Mean Dice        : {mean_dice:.4f}"
)

print(
    f"Minimum IoU      : {min_iou:.4f}"
)

print(
    f"Maximum IoU      : {max_iou:.4f}"
)

print(
    f"IoU >= 0.50      : {iou_50}/{len(results_df)}"
)

print(
    f"IoU >= 0.70      : {iou_70}/{len(results_df)}"
)

print("=" * 70)

print("\nSaved:")
print(csv_path)
print(summary_path)

print("=" * 70)