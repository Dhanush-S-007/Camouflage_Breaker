import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SINET_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

sys.path.insert(0, SINET_ROOT)

# --------------------------------------------------
# Paths
# --------------------------------------------------

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
        "Res2Net checkpoint not found."
    )

PRETRAINED_SINET = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth"
)

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "sinetv2"
)

os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "sinetv2_cod10k_best.pth"
)

# --------------------------------------------------
# Patch Res2Net loading
# --------------------------------------------------

import lib.Res2Net_v1b as res2net_module

_original_res2net = (
    res2net_module.res2net50_v1b_26w_4s
)


def local_res2net50_v1b_26w_4s(
    pretrained=False
):

    model = _original_res2net(
        pretrained=False
    )

    checkpoint = torch.load(
        RES2NET_CHECKPOINT,
        map_location="cpu"
    )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

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

from lib.Network_Res2Net_GRA_NCD import Network


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class COD10KDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=352,
        augment=False
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        self.mask_lookup = {}

        for filename in os.listdir(mask_dir):

            base = os.path.splitext(
                filename
            )[0]

            self.mask_lookup[base] = os.path.join(
                mask_dir,
                filename
            )

        self.samples = []

        for filename in self.images:

            base = os.path.splitext(
                filename
            )[0]

            if base in self.mask_lookup:

                self.samples.append(
                    filename
                )

        print(
            f"Dataset samples: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        filename = self.samples[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        base = os.path.splitext(
            filename
        )[0]

        mask_path = self.mask_lookup[base]

        image = cv2.imread(
            image_path
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Random horizontal flip
        if self.augment:

            if np.random.rand() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            # Random vertical flip
            if np.random.rand() < 0.2:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

        image = image.astype(
            np.float32
        ) / 255.0

        image = (
            image -
            np.array(
                [0.485, 0.456, 0.406],
                dtype=np.float32
            )
        )

        image = image / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = (
            mask.astype(
                np.float32
            ) / 255.0
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0).float()

        return image, mask


# --------------------------------------------------
# Loss
# --------------------------------------------------

def structure_loss(pred, mask):

    weit = 1 + 5 * torch.abs(
        torch.nn.functional.avg_pool2d(
            mask,
            kernel_size=31,
            stride=1,
            padding=15
        ) - mask
    )

    wbce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        mask,
        reduction="none"
    )

    wbce = (
        (weit * wbce).sum(
            dim=(2, 3)
        )
        /
        weit.sum(
            dim=(2, 3)
        )
    )

    pred_sigmoid = torch.sigmoid(pred)

    inter = (
        pred_sigmoid * mask * weit
    ).sum(
        dim=(2, 3)
    )

    union = (
        (
            pred_sigmoid + mask
        ) * weit
    ).sum(
        dim=(2, 3)
    )

    wiou = 1 - (
        inter + 1
    ) / (
        union - inter + 1
    )

    return (
        wbce + wiou
    ).mean()


# --------------------------------------------------
# Validation loss
# --------------------------------------------------

def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if isinstance(
                outputs,
                (list, tuple)
            ):
                prediction = outputs[-1]
            else:
                prediction = outputs

            loss = structure_loss(
                prediction,
                masks
            )

            total_loss += loss.item()

    return (
        total_loss /
        max(len(loader), 1)
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

print("=" * 70)
print("SINET-V2 FINE-TUNING ON COD10K")
print("=" * 70)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

if device.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("\nChecking dataset...")

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

print("Training images: OK")
print("Training masks : OK")

# --------------------------------------------------
# Dataset split
# --------------------------------------------------

full_dataset = COD10KDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    image_size=352,
    augment=True
)

total_size = len(
    full_dataset
)

val_size = int(
    total_size * 0.10
)

train_size = (
    total_size - val_size
)

generator = torch.Generator()

generator.manual_seed(42)

train_indices, val_indices = torch.utils.data.random_split(
    range(total_size),
    [train_size, val_size],
    generator=generator
)

train_indices = list(
    train_indices
)

val_indices = list(
    val_indices
)


class SubsetWithAugment(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        augment
    ):

        self.base_dataset = base_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        real_index = self.indices[index]

        filename = self.base_dataset.samples[
            real_index
        ]

        image_path = os.path.join(
            self.base_dataset.image_dir,
            filename
        )

        base = os.path.splitext(
            filename
        )[0]

        mask_path = self.base_dataset.mask_lookup[
            base
        ]

        image = cv2.imread(
            image_path
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (352, 352),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (352, 352),
            interpolation=cv2.INTER_NEAREST
        )

        if self.augment:

            if np.random.rand() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            if np.random.rand() < 0.2:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

        image = image.astype(
            np.float32
        ) / 255.0

        image = (
            image -
            np.array(
                [0.485, 0.456, 0.406],
                dtype=np.float32
            )
        )

        image = image / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = (
            mask.astype(
                np.float32
            ) / 255.0
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0).float()

        return image, mask


train_dataset = SubsetWithAugment(
    full_dataset,
    train_indices,
    augment=True
)

val_dataset = SubsetWithAugment(
    full_dataset,
    val_indices,
    augment=False
)

print(
    f"Training samples  : {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

batch_size = 8

if device.type == "cpu":
    batch_size = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda"
)

print(
    f"Batch size: {batch_size}"
)

# --------------------------------------------------
# Model
# --------------------------------------------------

print("\nLoading SINet-V2...")

model = Network(
    channel=32
)

checkpoint = torch.load(
    PRETRAINED_SINET,
    map_location="cpu"
)

if (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):
    checkpoint = checkpoint["state_dict"]

cleaned = {}

for key, value in checkpoint.items():

    if key.startswith("module."):
        key = key[7:]

    cleaned[key] = value

model.load_state_dict(
    cleaned,
    strict=False
)

model.to(device)

print(
    "Pretrained SINet-V2 loaded."
)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20
)

epochs = 20

best_val_loss = float("inf")

print("\nStarting training...")
print(
    f"Epochs: {epochs}"
)
print(
    f"Learning rate: 0.0001"
)

# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(
    1,
    epochs + 1
):

    model.train()

    train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{epochs}"
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

        if isinstance(
            outputs,
            (list, tuple)
        ):

            loss = 0.0

            for output in outputs:

                loss += structure_loss(
                    output,
                    masks
                )

            loss = loss / len(
                outputs
            )

        else:

            loss = structure_loss(
                outputs,
                masks
            )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= max(
        len(train_loader),
        1
    )

    val_loss = validate(
        model,
        val_loader,
        device
    )

    scheduler.step()

    print(
        f"\nEpoch {epoch}/{epochs}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"LR        : {optimizer.param_groups[0]['lr']:.8f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "✓ Best model saved."
        )

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: {best_val_loss:.4f}"
)

print(
    f"Best model:\n{BEST_MODEL_PATH}"
)

print("=" * 70)