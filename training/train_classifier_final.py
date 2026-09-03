import os
import sys
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIGURATION
# ============================================================

class Config:

    CROPS_ROOT = os.path.join(PROJECT_ROOT, "dataset", "crops")

    SPLIT = "Train"

    IMAGE_SIZE = 224

    BATCH_SIZE = 16

    NUM_WORKERS = 0

    EPOCHS = 30

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-5

    VAL_RATIO = 0.10

    SEED = 42

    NUM_CLASSES = 69

    CHECKPOINT_DIR = os.path.join(
        PROJECT_ROOT,
        "saved_models"
    )

    CHECKPOINT_NAME = "classifier_best.pth"

    CLASS_MAPPING_NAME = "class_mapping.json"

    PATIENCE = 5


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# RESNET50 CLASSIFIER
# ============================================================

class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=69):

        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# DATASET
# ============================================================

class CropDataset(Dataset):

    def __init__(
        self,
        crops_dir,
        split="Train",
        transform=None
    ):

        self.crops_dir = os.path.join(
            crops_dir,
            split
        )

        self.transform = transform

        self.samples = []

        self.class_to_idx = {}

        self.idx_to_class = {}

        self._load_labels()


    # --------------------------------------------------------
    # LOAD COD10K LABELS
    # --------------------------------------------------------

    def _load_labels(self):

        txt_path = os.path.join(

            PROJECT_ROOT,

            "dataset",

            self.split_name(),

            f"CAM-NonCAM_Instance_{self.split_name()}.txt"
        )

        if not os.path.exists(txt_path):

            raise FileNotFoundError(
                f"Label file not found:\n{txt_path}"
            )


        # ----------------------------------------------------
        # Read TXT
        # ----------------------------------------------------

        with open(
            txt_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()


        label_map = {}

        current_filename = None

        current_category = None


        # ----------------------------------------------------
        # Parse COD10K TXT
        # ----------------------------------------------------

        i = 0

        while i < len(lines):

            line = lines[i].strip()


            if line.startswith("[INFO]"):

                parts = line.split()

                if len(parts) >= 3:

                    filename = parts[1]

                    cam_flag = parts[2]

                    base = os.path.splitext(
                        filename
                    )[0]


                    # Next line contains class path

                    if i + 1 < len(lines):

                        class_line = lines[i + 1].strip()

                        if class_line:

                            class_path = class_line.split()[0]

                            category = class_path.split("/")[-1]

                            label_map[base] = category


                            i += 2

                            continue


            i += 1


        if len(label_map) == 0:

            raise RuntimeError(
                "No labels were loaded from COD10K TXT file."
            )


        # ----------------------------------------------------
        # Create stable class mapping
        # ----------------------------------------------------

        all_classes = sorted(
            set(label_map.values())
        )


        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(all_classes)
        }


        self.idx_to_class = {
            str(index): class_name
            for class_name, index
            in self.class_to_idx.items()
        }


        # ----------------------------------------------------
        # Create crop samples
        # ----------------------------------------------------

        if not os.path.exists(self.crops_dir):

            raise FileNotFoundError(
                f"Crop directory not found:\n{self.crops_dir}"
            )


        for filename in sorted(
            os.listdir(self.crops_dir)
        ):

            if not filename.endswith("_crop.jpg"):

                continue


            base = filename.replace(
                "_crop.jpg",
                ""
            )


            if base not in label_map:

                continue


            class_name = label_map[base]

            class_index = self.class_to_idx[
                class_name
            ]


            image_path = os.path.join(
                self.crops_dir,
                filename
            )


            self.samples.append(
                (
                    image_path,
                    class_index
                )
            )


        print(
            f"[INFO] {self.split_name()} "
            f"classes: {len(self.class_to_idx)}"
        )

        print(
            f"[INFO] {self.split_name()} "
            f"crop samples: {len(self.samples)}"
        )


    def split_name(self):

        return os.path.basename(
            self.crops_dir
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, label = self.samples[index]


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)


        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform():

    return transforms.Compose([

        transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(

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
    ])


def get_val_transform():

    return transforms.Compose([

        transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

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
    ])


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    correct = 0

    total = 0


    progress = tqdm(
        loader,
        desc="Training"
    )


    for images, labels in progress:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        accuracy = (
            100.0 * correct / total
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{accuracy:.2f}%"
        )


    return (
        total_loss / len(loader),
        100.0 * correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0


    progress = tqdm(
        loader,
        desc="Validation"
    )


    for images, labels in progress:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        total_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    return (
        total_loss / len(loader),
        100.0 * correct / total
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_accuracy,
    class_to_idx,
    idx_to_class
):

    os.makedirs(
        Config.CHECKPOINT_DIR,
        exist_ok=True
    )


    checkpoint_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.CHECKPOINT_NAME
    )


    checkpoint = {

        "epoch": epoch,

        "best_accuracy": best_accuracy,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "class_to_idx":
            class_to_idx,

        "idx_to_class":
            idx_to_class,

        "num_classes":
            Config.NUM_CLASSES
    }


    torch.save(
        checkpoint,
        checkpoint_path
    )


    # Also save class mapping separately

    mapping_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.CLASS_MAPPING_NAME
    )


    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            idx_to_class,
            f,
            indent=4
        )


    print(
        f"[OK] Model saved: {checkpoint_path}"
    )

    print(
        f"[OK] Class mapping saved: {mapping_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CAMOUFLAGE BREAKER - RESNET50 CLASSIFIER"
    )

    print("=" * 70)


    config = Config()


    set_seed(
        config.SEED
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"[INFO] Device: {device}"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\n[INFO] Loading crop dataset..."
    )


    full_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=None
    )


    total_samples = len(
        full_dataset
    )


    if total_samples == 0:

        raise RuntimeError(
            "No crop images found."
        )


    print(
        f"[INFO] Total samples: "
        f"{total_samples}"
    )


    print(
        f"[INFO] Number of classes: "
        f"{len(full_dataset.class_to_idx)}"
    )


    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    if len(full_dataset.class_to_idx) != config.NUM_CLASSES:

        raise RuntimeError(

            f"Expected {config.NUM_CLASSES} classes, "

            f"but found "
            f"{len(full_dataset.class_to_idx)}."
        )


    # --------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------

    val_size = int(
        total_samples *
        config.VAL_RATIO
    )


    train_size = (
        total_samples -
        val_size
    )


    train_subset, val_subset = random_split(

        full_dataset,

        [train_size, val_size],

        generator=torch.Generator().manual_seed(
            config.SEED
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Create separate dataset objects
    # so train and validation transforms
    # do not overwrite each other.
    # --------------------------------------------------------

    train_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=get_train_transform()
    )


    val_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=get_val_transform()
    )


    train_dataset.samples = [
        train_dataset.samples[i]
        for i in train_subset.indices
    ]


    val_dataset.samples = [
        val_dataset.samples[i]
        for i in val_subset.indices
    ]


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    print(
        f"[INFO] Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"[INFO] Validation samples: "
        f"{len(val_dataset)}"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n[INFO] Building ResNet50..."
    )


    model = ResNet50Classifier(

        num_classes=config.NUM_CLASSES
    )


    model = model.to(device)


    total_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    print(
        f"[INFO] Parameters: "
        f"{total_parameters:,}"
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = 0.0

    patience_counter = 0


    print(
        "\n[INFO] Starting training..."
    )


    print(
        "-" * 70
    )


    for epoch in range(
        config.EPOCHS
    ):

        start_time = time.time()


        train_loss, train_accuracy = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device
        )


        val_loss, val_accuracy = validate(

            model,

            val_loader,

            criterion,

            device
        )


        scheduler.step(
            val_accuracy
        )


        elapsed = (
            time.time()
            - start_time
        )


        print()

        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{config.EPOCHS}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            patience_counter = 0


            save_checkpoint(

                model,

                optimizer,

                epoch,

                best_accuracy,

                full_dataset.class_to_idx,

                full_dataset.idx_to_class
            )


            print(
                f"[★] New best model!"
            )

        else:

            patience_counter += 1

            print(
                f"[INFO] No improvement "
                f"({patience_counter}/"
                f"{config.PATIENCE})"
            )


        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= config.PATIENCE:

            print(
                "\n[STOP] Early stopping."
            )

            break


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "CLASSIFIER TRAINING COMPLETE"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()