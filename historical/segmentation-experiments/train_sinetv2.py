import os
import sys
import random
import importlib
import numpy as np
import cv2

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SINET_BASE = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2"
)


# ============================================================
# FIND SINET-V2 SOURCE AUTOMATICALLY
# ============================================================

SOURCE_ROOT = None
NETWORK_FILE = None
RES2NET_FILE = None

for root, dirs, files in os.walk(SINET_BASE):

    if (
        "Network_Res2Net_GRA_NCD.py" in files
        and
        "Res2Net_v1b.py" in files
    ):
        SOURCE_ROOT = root
        NETWORK_FILE = os.path.join(
            root,
            "Network_Res2Net_GRA_NCD.py"
        )
        RES2NET_FILE = os.path.join(
            root,
            "Res2Net_v1b.py"
        )
        break


if SOURCE_ROOT is None:

    raise FileNotFoundError(
        "\nSINet-V2 source was not found.\n"
        "Expected official files:\n"
        "Network_Res2Net_GRA_NCD.py\n"
        "Res2Net_v1b.py\n"
    )


# Python must see the folder containing "lib"
SOURCE_PARENT = os.path.dirname(
    SOURCE_ROOT
)

if SOURCE_PARENT not in sys.path:
    sys.path.insert(
        0,
        SOURCE_PARENT
    )

if SOURCE_ROOT not in sys.path:
    sys.path.insert(
        0,
        SOURCE_ROOT
    )


# ============================================================
# IMPORT OFFICIAL SINET-V2 MODULES
# ============================================================

res2net_module = importlib.import_module(
    "lib.Res2Net_v1b"
)


# ============================================================
# FIND RES2NET CHECKPOINT
# ============================================================

RES2NET_CHECKPOINT = None

for root, dirs, files in os.walk(SINET_BASE):

    if (
        "res2net50_v1b_26w_4s-3cf99910.pth"
        in files
    ):

        RES2NET_CHECKPOINT = os.path.join(
            root,
            "res2net50_v1b_26w_4s-3cf99910.pth"
        )

        break


if RES2NET_CHECKPOINT is None:

    raise FileNotFoundError(
        "\nRes2Net checkpoint was not found.\n"
        "Expected:\n"
        "res2net50_v1b_26w_4s-3cf99910.pth\n"
    )


# ============================================================
# PATCH RES2NET LOADING
# ============================================================

original_res2net_function = (
    res2net_module.res2net50_v1b_26w_4s
)


def local_res2net50_v1b_26w_4s(
    pretrained=True
):

    model = original_res2net_function(
        pretrained=False
    )

    if pretrained:

        checkpoint = torch.load(
            RES2NET_CHECKPOINT,
            map_location="cpu"
        )

        if isinstance(
            checkpoint,
            dict
        ):

            if "state_dict" in checkpoint:
                checkpoint = checkpoint[
                    "state_dict"
                ]

            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint[
                    "model_state_dict"
                ]

        cleaned = {}

        for key, value in checkpoint.items():

            if key.startswith("module."):
                key = key[7:]

            cleaned[key] = value

        model.load_state_dict(
            cleaned,
            strict=False
        )

    return model


res2net_module.res2net50_v1b_26w_4s = (
    local_res2net50_v1b_26w_4s
)


# ============================================================
# IMPORT SINET-V2 NETWORK AFTER PATCH
# ============================================================

network_module = importlib.import_module(
    "lib.Network_Res2Net_GRA_NCD"
)

Network = network_module.Network


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 352

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.10

SEED = 42


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Train",
    "Image"
)

TRAIN_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Train",
    "GT_Object"
)


# ============================================================
# PRETRAINED SINET-V2
# ============================================================

PRETRAINED_SINET = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth"
)


# ============================================================
# OUTPUT
# ============================================================

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "sinetv2"
)

BEST_MODEL = os.path.join(
    SAVE_DIR,
    "sinetv2_cod10k_best.pth"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class COD10KDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        indices=None,
        augment=False
    ):

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.augment = augment

        image_files = sorted([
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ])

        pairs = []

        for image_name in image_files:

            base = os.path.splitext(
                image_name
            )[0]

            mask_name = None

            for extension in [
                ".png",
                ".jpg",
                ".jpeg"
            ]:

                candidate = (
                    base + extension
                )

                candidate_path = os.path.join(
                    mask_dir,
                    candidate
                )

                if os.path.exists(
                    candidate_path
                ):

                    mask_name = candidate

                    break

            if mask_name is not None:

                pairs.append(
                    (
                        image_name,
                        mask_name
                    )
                )

        if indices is not None:

            pairs = [
                pairs[i]
                for i in indices
            ]

        self.pairs = pairs

        self.normalize = transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )

    def __len__(self):

        return len(self.pairs)

    def __getitem__(
        self,
        index
    ):

        image_name, mask_name = (
            self.pairs[index]
        )

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = cv2.imread(
            image_path
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        if mask is None:

            raise RuntimeError(
                f"Cannot read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

            if random.random() < 0.3:

                angle = random.choice(
                    [
                        -10,
                        -5,
                        5,
                        10
                    ]
                )

                center = (
                    IMAGE_SIZE // 2,
                    IMAGE_SIZE // 2
                )

                matrix = cv2.getRotationMatrix2D(
                    center,
                    angle,
                    1.0
                )

                image = cv2.warpAffine(
                    image,
                    matrix,
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    borderMode=cv2.BORDER_REFLECT
                )

                mask = cv2.warpAffine(
                    mask,
                    matrix,
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT
                )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )

        mask = (
            mask.astype(
                np.float32
            )
            /
            255.0
        )

        image = torch.from_numpy(
            image.transpose(
                2,
                0,
                1
            )
        )

        image = self.normalize(
            image
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        mask = (
            mask > 0.5
        ).float()

        return image, mask


# ============================================================
# STRUCTURE LOSS
# ============================================================

def structure_loss(
    prediction,
    target
):

    if prediction.shape[-2:] != target.shape[-2:]:

        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    weight = (
        1
        +
        5
        *
        torch.abs(
            F.avg_pool2d(
                target,
                kernel_size=31,
                stride=1,
                padding=15
            )
            -
            target
        )
    )

    bce = F.binary_cross_entropy_with_logits(
        prediction,
        target,
        reduction="none"
    )

    bce = (
        weight * bce
    ).sum(
        dim=(2, 3)
    ) / weight.sum(
        dim=(2, 3)
    )

    prediction_sigmoid = torch.sigmoid(
        prediction
    )

    intersection = (
        prediction_sigmoid
        *
        target
        *
        weight
    ).sum(
        dim=(2, 3)
    )

    union = (
        prediction_sigmoid
        +
        target
    ).mul(
        weight
    ).sum(
        dim=(2, 3)
    )

    iou_loss = 1 - (
        (intersection + 1)
        /
        (union - intersection + 1)
    )

    return (
        bce + iou_loss
    ).mean()


def calculate_loss(
    outputs,
    target
):

    if isinstance(
        outputs,
        (tuple, list)
    ):

        losses = []

        for output in outputs:

            losses.append(
                structure_loss(
                    output,
                    target
                )
            )

        return sum(losses) / len(losses)

    return structure_loss(
        outputs,
        target
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = calculate_loss(
                outputs,
                masks
            )

            total_loss += (
                loss.item()
            )

            batches += 1

    return (
        total_loss
        /
        max(
            batches,
            1
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("SINET-V2 COD10K FINE-TUNING")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU Memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                /
                (1024 ** 3),
                2
            ),
            "GB"
        )

    print()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print("Checking files...")

    print(
        "SINet-V2 source:",
        SOURCE_ROOT
    )

    print(
        "Res2Net checkpoint:",
        RES2NET_CHECKPOINT
    )

    print(
        "SINet-V2 checkpoint:",
        PRETRAINED_SINET
    )

    print()

    if not os.path.exists(
        TRAIN_IMAGE_DIR
    ):

        raise FileNotFoundError(
            TRAIN_IMAGE_DIR
        )

    if not os.path.exists(
        TRAIN_MASK_DIR
    ):

        raise FileNotFoundError(
            TRAIN_MASK_DIR
        )

    if not os.path.exists(
        PRETRAINED_SINET
    ):

        raise FileNotFoundError(
            PRETRAINED_SINET
        )

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------

    image_count = len(
        os.listdir(
            TRAIN_IMAGE_DIR
        )
    )

    mask_count = len(
        os.listdir(
            TRAIN_MASK_DIR
        )
    )

    print(
        "Training images:",
        image_count
    )

    print(
        "Training masks :",
        mask_count
    )

    if image_count != 6000:

        raise RuntimeError(
            f"Expected 6000 training images, found {image_count}"
        )

    if mask_count != 6000:

        raise RuntimeError(
            f"Expected 6000 training masks, found {mask_count}"
        )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("Building dataset...")

    base_dataset = COD10KDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    total = len(
        base_dataset
    )

    validation_size = int(
        total * VAL_RATIO
    )

    training_size = (
        total
        -
        validation_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_subset, val_subset = random_split(
        range(total),
        [
            training_size,
            validation_size
        ],
        generator=generator
    )

    train_indices = (
        train_subset.indices
    )

    val_indices = (
        val_subset.indices
    )

    train_dataset = COD10KDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        indices=train_indices,
        augment=True
    )

    val_dataset = COD10KDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        indices=val_indices,
        augment=False
    )

    print(
        "Total:",
        total
    )

    print(
        "Training:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    print()

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Creating SINet-V2 model...")

    model = Network(
        channel=32
    )

    print(
        "Loading pretrained SINet-V2 weights..."
    )

    checkpoint = torch.load(
        PRETRAINED_SINET,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint[
                "model_state_dict"
            ]

    cleaned = {}

    for key, value in checkpoint.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model = model.to(
        device
    )

    print(
        "✓ SINet-V2 loaded successfully."
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_loss = float(
        "inf"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("=" * 70)
    print("STARTING COD10K TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{EPOCHS}"
            )
        )

        for images, masks in progress:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images
            )

            loss = calculate_loss(
                outputs,
                masks
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )

        train_loss = (
            running_loss
            /
            len(train_loader)
        )

        val_loss = validate(
            model,
            val_loader,
            device
        )

        scheduler.step()

        current_lr = (
            scheduler.get_last_lr()[0]
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

        print(
            f"Learning Rate : {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss
                },
                BEST_MODEL
            )

            print(
                "✓ BEST MODEL SAVED"
            )

            print(
                BEST_MODEL
            )

        print(
            "-" * 70
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Model saved at:"
    )

    print(
        BEST_MODEL
    )


if __name__ == "__main__":

    main()