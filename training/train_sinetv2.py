import os
import sys
import random
import importlib.util

import cv2
import numpy as np
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

SINET_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

SINET_LIB = os.path.join(
    SINET_ROOT,
    "lib"
)

if not os.path.isdir(SINET_ROOT):
    raise FileNotFoundError(
        f"SINet-V2 source folder not found:\n{SINET_ROOT}"
    )

if not os.path.isdir(SINET_LIB):
    raise FileNotFoundError(
        f"SINet-V2 lib folder not found:\n{SINET_LIB}"
    )


# ============================================================
# CREATE LIB PACKAGE
# ============================================================

import types

lib_package = types.ModuleType("lib")
lib_package.__path__ = [SINET_LIB]

sys.modules["lib"] = lib_package


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{file_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD RES2NET
# ============================================================

RES2NET_FILE = os.path.join(
    SINET_LIB,
    "Res2Net_v1b.py"
)

if not os.path.isfile(RES2NET_FILE):
    raise FileNotFoundError(
        f"Res2Net source not found:\n{RES2NET_FILE}"
    )

res2net_module = load_module(
    "lib.Res2Net_v1b",
    RES2NET_FILE
)

lib_package.Res2Net_v1b = res2net_module


# ============================================================
# FIND RES2NET CHECKPOINT
# ============================================================

RES2NET_CHECKPOINT = None

for root, dirs, files in os.walk(
    os.path.join(
        PROJECT_ROOT,
        "models",
        "sinetv2"
    )
):

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
        "Res2Net checkpoint not found."
    )


# ============================================================
# PATCH RES2NET LOADING
# ============================================================

original_res2net_function = (
    res2net_module.res2net50_v1b_26w_4s
)


def local_res2net50_v1b_26w_4s(
    pretrained=True,
    **kwargs
):

    model = original_res2net_function(
        pretrained=False,
        **kwargs
    )

    if pretrained:

        checkpoint = torch.load(
            RES2NET_CHECKPOINT,
            map_location="cpu"
        )

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

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
# LOAD SINET-V2 NETWORK
# ============================================================

NETWORK_FILE = os.path.join(
    SINET_LIB,
    "Network_Res2Net_GRA_NCD.py"
)

if not os.path.isfile(NETWORK_FILE):
    raise FileNotFoundError(
        f"SINet-V2 network source not found:\n{NETWORK_FILE}"
    )

network_module = load_module(
    "lib.Network_Res2Net_GRA_NCD",
    NETWORK_FILE
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

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

BEST_MODEL = os.path.join(
    SAVE_DIR,
    "sinetv2_cod10k_best.pth"
)


# ============================================================
# SEED
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
                (".jpg", ".jpeg", ".png")
            )
        ])

        mask_files = sorted([
            f
            for f in os.listdir(mask_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        mask_lookup = {}

        for filename in mask_files:

            base = os.path.splitext(
                filename
            )[0]

            mask_lookup[base] = filename

        pairs = []

        for filename in image_files:

            base = os.path.splitext(
                filename
            )[0]

            if base in mask_lookup:

                pairs.append(
                    (
                        filename,
                        mask_lookup[base]
                    )
                )

        if len(pairs) == 0:

            raise RuntimeError(
                "No image-mask pairs were found."
            )

        if indices is not None:

            pairs = [
                pairs[i]
                for i in indices
            ]

        self.files = pairs

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

        return len(self.files)

    def __getitem__(self, index):

        image_name, mask_name = (
            self.files[index]
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
                f"Cannot read image:\n{image_path}"
            )

        if mask is None:

            raise RuntimeError(
                f"Cannot read mask:\n{mask_path}"
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

                angle = random.choice([
                    -10,
                    -5,
                    5,
                    10
                ])

                matrix = cv2.getRotationMatrix2D(
                    (
                        IMAGE_SIZE // 2,
                        IMAGE_SIZE // 2
                    ),
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

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        mask = (
            mask.astype(
                np.float32
            )
            / 255.0
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
    pred,
    mask
):

    if pred.shape[-2:] != mask.shape[-2:]:

        pred = F.interpolate(
            pred,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    weit = (
        1
        +
        5
        *
        torch.abs(
            F.avg_pool2d(
                mask,
                kernel_size=31,
                stride=1,
                padding=15
            )
            -
            mask
        )
    )

    wbce = F.binary_cross_entropy_with_logits(
        pred,
        mask,
        reduction="none"
    )

    wbce = (
        (
            weit * wbce
        ).sum(
            dim=(2, 3)
        )
        /
        weit.sum(
            dim=(2, 3)
        )
    )

    pred_sigmoid = torch.sigmoid(
        pred
    )

    inter = (
        pred_sigmoid
        *
        mask
        *
        weit
    ).sum(
        dim=(2, 3)
    )

    union = (
        (
            pred_sigmoid
            +
            mask
        )
        *
        weit
    ).sum(
        dim=(2, 3)
    )

    wiou = 1 - (
        (inter + 1)
        /
        (union - inter + 1)
    )

    return (
        wbce + wiou
    ).mean()


def calculate_loss(
    outputs,
    mask
):

    if isinstance(
        outputs,
        (tuple, list)
    ):

        losses = []

        for output in outputs:

            if output is not None:

                losses.append(
                    structure_loss(
                        output,
                        mask
                    )
                )

        if not losses:

            raise RuntimeError(
                "SINet-V2 returned no valid output."
            )

        return sum(losses) / len(losses)

    return structure_loss(
        outputs,
        mask
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

            total_loss += loss.item()

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                /
                (1024 ** 3),
                2
            ),
            "GB"
        )

    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print(
        "Checking dataset..."
    )

    if not os.path.isdir(
        TRAIN_IMAGE_DIR
    ):

        raise FileNotFoundError(
            TRAIN_IMAGE_DIR
        )

    if not os.path.isdir(
        TRAIN_MASK_DIR
    ):

        raise FileNotFoundError(
            TRAIN_MASK_DIR
        )

    image_count = len([
        f
        for f in os.listdir(
            TRAIN_IMAGE_DIR
        )
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    mask_count = len([
        f
        for f in os.listdir(
            TRAIN_MASK_DIR
        )
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    print(
        "Training images:",
        image_count
    )

    print(
        "Training masks :",
        mask_count
    )

    print()

    # --------------------------------------------------------
    # Check pretrained files
    # --------------------------------------------------------

    print(
        "Checking pretrained weights..."
    )

    if not os.path.isfile(
        PRETRAINED_SINET
    ):

        raise FileNotFoundError(
            PRETRAINED_SINET
        )

    if not os.path.isfile(
        RES2NET_CHECKPOINT
    ):

        raise FileNotFoundError(
            RES2NET_CHECKPOINT
        )

    print(
        "SINet-V2 checkpoint: OK"
    )

    print(
        "Res2Net checkpoint : OK"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "Creating dataset..."
    )

    base_dataset = COD10KDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    total = len(
        base_dataset
    )

    val_size = int(
        total * VAL_RATIO
    )

    train_size = (
        total
        -
        val_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_subset, val_subset = random_split(
        range(total),
        [
            train_size,
            val_size
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

    print(
        "Loading SINet-V2..."
    )

    model = Network(
        channel=32
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

    cleaned_checkpoint = {}

    for key, value in checkpoint.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned_checkpoint[
            key
        ] = value

    model.load_state_dict(
        cleaned_checkpoint,
        strict=True
    )

    model = model.to(
        device
    )

    print(
        "SINet-V2 loaded successfully."
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
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}"
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

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
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
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss  : {val_loss:.6f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

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
        "Best model:",
        BEST_MODEL
    )


if __name__ == "__main__":
    main()