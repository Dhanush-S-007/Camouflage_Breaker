import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sam2",
    "sam2.1_hiera_small.pt"
)

CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

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

NUM_IMAGES = 20


def calculate_iou(pred, gt):

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


def calculate_dice(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    total = pred.sum() + gt.sum()

    if total == 0:
        return 0.0

    return (2 * intersection) / total


def find_ground_truth(image_name):

    base_name = os.path.splitext(image_name)[0]

    for file_name in os.listdir(MASK_DIR):

        file_base = os.path.splitext(file_name)[0]

        if file_base == base_name:
            return os.path.join(
                MASK_DIR,
                file_name
            )

    return None


def evaluate_selection(
    masks,
    gt,
    strategy
):

    if strategy == "largest_area":

        selected = max(
            masks,
            key=lambda x: x["area"]
        )

    elif strategy == "predicted_iou":

        selected = max(
            masks,
            key=lambda x: x["predicted_iou"]
        )

    elif strategy == "stability_score":

        selected = max(
            masks,
            key=lambda x: x["stability_score"]
        )

    mask = selected["segmentation"]

    iou = calculate_iou(
        mask,
        gt
    )

    dice = calculate_dice(
        mask,
        gt
    )

    return iou, dice


def main():

    print("=" * 65)
    print("SAM 2.1 AUTOMATIC MASK SELECTION BENCHMARK")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    selected_images = image_files[:NUM_IMAGES]

    print(
        f"Images selected: "
        f"{len(selected_images)}"
    )

    print()
    print("Loading SAM 2.1...")

    model = build_sam2(
        CONFIG,
        CHECKPOINT,
        device=device
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        points_per_batch=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100
    )

    print("SAM 2.1 loaded.")
    print()

    strategies = [
        "largest_area",
        "predicted_iou",
        "stability_score"
    ]

    scores = {
        strategy: {
            "iou": [],
            "dice": []
        }
        for strategy in strategies
    }

    for image_name in tqdm(
        selected_images,
        desc="Evaluating"
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        gt_path = find_ground_truth(
            image_name
        )

        if gt_path is None:
            continue

        image = cv2.imread(
            image_path
        )

        gt = cv2.imread(
            gt_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or gt is None:
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        gt = gt > 127

        masks = mask_generator.generate(
            image_rgb
        )

        if len(masks) == 0:
            continue

        for strategy in strategies:

            iou, dice = evaluate_selection(
                masks,
                gt,
                strategy
            )

            scores[strategy]["iou"].append(
                iou
            )

            scores[strategy]["dice"].append(
                dice
            )

    print()
    print("=" * 65)
    print("AUTOMATIC MASK SELECTION RESULTS")
    print("=" * 65)

    for strategy in strategies:

        iou_values = scores[strategy]["iou"]
        dice_values = scores[strategy]["dice"]

        mean_iou = np.mean(iou_values)
        mean_dice = np.mean(dice_values)

        strong_50 = sum(
            iou >= 0.50
            for iou in iou_values
        )

        strong_70 = sum(
            iou >= 0.70
            for iou in iou_values
        )

        print()
        print(
            f"Strategy: {strategy}"
        )

        print(
            f"Mean IoU : {mean_iou:.4f}"
        )

        print(
            f"Mean Dice: {mean_dice:.4f}"
        )

        print(
            f"IoU >= 0.50: "
            f"{strong_50}/{len(iou_values)}"
        )

        print(
            f"IoU >= 0.70: "
            f"{strong_70}/{len(iou_values)}"
        )

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()