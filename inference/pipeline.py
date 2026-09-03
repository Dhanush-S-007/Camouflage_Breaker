# ============================================================
# Camouflage Breaker - Complete Inference Pipeline
# ============================================================

import os
import sys
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from models.resunet import ResUNet


class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=69):

        super().__init__()

        self.backbone = models.resnet50(
            weights=None
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


class CamouflageBreakerPipeline:

    def __init__(
        self,
        seg_model_path=None,
        classifier_model_path=None,
        class_mapping_path=None
    ):

        # ----------------------------------------------------
        # Default paths
        # ----------------------------------------------------

        if seg_model_path is None:

            seg_model_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "resunet_best.pth"
            )

        if classifier_model_path is None:

            classifier_model_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "classifier_best.pth"
            )

        if class_mapping_path is None:

            class_mapping_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "class_mapping.json"
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("=" * 70)
        print("CAMOUFLAGE BREAKER INFERENCE PIPELINE")
        print("=" * 70)

        print(
            f"Using device: {self.device}"
        )

        # ----------------------------------------------------
        # Load ResUNet
        # ----------------------------------------------------

        print(
            "\nLoading ResUNet segmentation model..."
        )

        if not os.path.exists(
            seg_model_path
        ):

            raise FileNotFoundError(
                f"ResUNet model not found:\n"
                f"{seg_model_path}"
            )

        self.seg_model = ResUNet(
            encoder_name="resnet50",
            encoder_weights=None
        )

        seg_checkpoint = torch.load(
            seg_model_path,
            map_location=self.device
        )

        if (
            isinstance(seg_checkpoint, dict)
            and
            "model_state_dict" in seg_checkpoint
        ):

            self.seg_model.load_state_dict(
                seg_checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            self.seg_model.load_state_dict(
                seg_checkpoint
            )

        self.seg_model.to(
            self.device
        )

        self.seg_model.eval()

        print(
            "✓ ResUNet loaded"
        )

        # ----------------------------------------------------
        # Load Classifier Checkpoint
        # ----------------------------------------------------

        print(
            "\nLoading trained ResNet50 classifier..."
        )

        if not os.path.exists(
            classifier_model_path
        ):

            raise FileNotFoundError(
                f"Classifier model not found:\n"
                f"{classifier_model_path}"
            )

        checkpoint = torch.load(
            classifier_model_path,
            map_location=self.device
        )

        # ----------------------------------------------------
        # Read number of classes
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and
            "num_classes" in checkpoint
        ):

            self.num_classes = int(
                checkpoint["num_classes"]
            )

        else:

            self.num_classes = 69

        # ----------------------------------------------------
        # Create EXACT same classifier architecture
        # used during Colab training
        # ----------------------------------------------------

        self.classifier = ResNet50Classifier(
            num_classes=self.num_classes
        )

        # ----------------------------------------------------
        # Load state dictionary
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and
            "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        self.classifier.load_state_dict(
            state_dict
        )

        self.classifier.to(
            self.device
        )

        self.classifier.eval()

        print(
            f"✓ ResNet50 classifier loaded "
            f"({self.num_classes} classes)"
        )

        # ----------------------------------------------------
        # Load class mapping
        # ----------------------------------------------------

        print(
            "\nLoading class mapping..."
        )

        if not os.path.exists(
            class_mapping_path
        ):

            raise FileNotFoundError(
                f"Class mapping not found:\n"
                f"{class_mapping_path}"
            )

        with open(
            class_mapping_path,
            "r",
            encoding="utf-8"
        ) as f:

            mapping = json.load(f)

        self.idx_to_class = mapping[
            "idx_to_class"
        ]

        self.class_to_idx = mapping[
            "class_to_idx"
        ]

        print(
            f"✓ Loaded "
            f"{len(self.idx_to_class)} class names"
        )

        # ----------------------------------------------------
        # Classification preprocessing
        # ----------------------------------------------------

        self.cls_transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize(
                (224, 224)
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

        print(
            "\n✓ Pipeline ready"
        )

        print("=" * 70)

    # ========================================================
    # PREPROCESS IMAGE FOR RESUNET
    # ========================================================

    def preprocess_image(
        self,
        image
    ):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_resized = cv2.resize(
            image_rgb,
            (352, 352)
        )

        image_float = (
            image_resized.astype(
                np.float32
            ) / 255.0
        )

        image_tensor = torch.from_numpy(
            image_float
        )

        image_tensor = image_tensor.permute(
            2,
            0,
            1
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        image_tensor = (
            image_tensor - mean
        ) / std

        image_tensor = image_tensor.unsqueeze(
            0
        )

        return image_tensor.to(
            self.device
        )

    # ========================================================
    # SEGMENTATION
    # ========================================================

    def get_mask(
        self,
        image_tensor,
        threshold=0.5
    ):

        with torch.no_grad():

            output = self.seg_model(
                image_tensor
            )

            probability = torch.sigmoid(
                output
            )

            mask = (
                probability
                .squeeze()
                .cpu()
                .numpy()
            )

        binary_mask = (
            mask >= threshold
        ).astype(
            np.uint8
        )

        return binary_mask

    # ========================================================
    # RESIZE MASK
    # ========================================================

    def resize_mask(
        self,
        mask,
        image
    ):

        return cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # ========================================================
    # BOUNDARY
    # ========================================================

    def draw_boundary(
        self,
        image,
        mask,
        thickness=3
    ):

        result = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(
                contour
            ) > 20:

                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    (0, 0, 255),
                    thickness
                )

        return result

    # ========================================================
    # COLORED OVERLAY
    # ========================================================

    def create_overlay(
        self,
        image,
        mask,
        alpha=0.35
    ):

        colored_mask = np.zeros_like(
            image
        )

        colored_mask[
            mask > 0
        ] = (
            0,
            0,
            255
        )

        result = cv2.addWeighted(
            image,
            1 - alpha,
            colored_mask,
            alpha,
            0
        )

        return result

    # ========================================================
    # CROP OBJECT
    # ========================================================

    def crop_object(
        self,
        image,
        mask,
        padding=20
    ):

        coordinates = np.where(
            mask > 0
        )

        if len(
            coordinates[0]
        ) == 0:

            return None

        y_min = coordinates[0].min()
        y_max = coordinates[0].max()

        x_min = coordinates[1].min()
        x_max = coordinates[1].max()

        height, width = image.shape[:2]

        y_min = max(
            0,
            y_min - padding
        )

        y_max = min(
            height,
            y_max + padding + 1
        )

        x_min = max(
            0,
            x_min - padding
        )

        x_max = min(
            width,
            x_max + padding + 1
        )

        crop = image[
            y_min:y_max,
            x_min:x_max
        ]

        if crop.size == 0:

            return None

        return crop

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify_object(
        self,
        crop
    ):

        if crop is None:

            return (
                "No object detected",
                0.0,
                None
            )

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        image_tensor = self.cls_transform(
            crop_rgb
        )

        image_tensor = image_tensor.unsqueeze(
            0
        ).to(
            self.device
        )

        with torch.no_grad():

            outputs = self.classifier(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        predicted_index = (
            predicted_index.item()
        )

        confidence = (
            confidence.item() * 100
        )

        class_name = self.idx_to_class.get(
            str(predicted_index),
            f"Class_{predicted_index + 1}"
        )

        return (
            class_name,
            confidence,
            predicted_index
        )

    # ========================================================
    # COMPLETE PREDICTION
    # ========================================================

    def predict(
        self,
        image,
        threshold=0.5
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        if isinstance(
            image,
            str
        ):

            image_path = image

            image = cv2.imread(
                image_path
            )

            if image is None:

                raise ValueError(
                    f"Could not load image:\n"
                    f"{image_path}"
                )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original = image.copy()

        # ----------------------------------------------------
        # ResUNet
        # ----------------------------------------------------

        image_tensor = self.preprocess_image(
            image
        )

        mask_small = self.get_mask(
            image_tensor,
            threshold=threshold
        )

        mask = self.resize_mask(
            mask_small,
            image
        )

        # ----------------------------------------------------
        # Check detection
        # ----------------------------------------------------

        object_pixels = np.sum(
            mask > 0
        )

        if object_pixels == 0:

            return {

                "original":
                    original,

                "mask":
                    mask,

                "boundary":
                    original.copy(),

                "overlay":
                    original.copy(),

                "crop":
                    None,

                "class_name":
                    "No object detected",

                "confidence":
                    0.0,

                "predicted_index":
                    None,

                "object_detected":
                    False
            }

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        boundary = self.draw_boundary(
            original,
            mask
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = self.create_overlay(
            original,
            mask
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        crop = self.crop_object(
            original,
            mask
        )

        # ----------------------------------------------------
        # ResNet50
        # ----------------------------------------------------

        (
            class_name,
            confidence,
            predicted_index
        ) = self.classify_object(
            crop
        )

        return {

            "original":
                original,

            "mask":
                mask,

            "boundary":
                boundary,

            "overlay":
                overlay,

            "crop":
                crop,

            "class_name":
                class_name,

            "confidence":
                confidence,

            "predicted_index":
                predicted_index,

            "object_detected":
                True
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("TESTING CAMOUFLAGE BREAKER")
    print("=" * 70)

    pipeline = CamouflageBreakerPipeline()

    test_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "Test",
        "Image"
    )

    if not os.path.exists(
        test_dir
    ):

        print(
            f"\nTest folder not found:\n"
            f"{test_dir}"
        )

        sys.exit()

    test_images = [

        os.path.join(
            test_dir,
            filename
        )

        for filename in sorted(
            os.listdir(test_dir)
        )

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    if len(test_images) == 0:

        print(
            "\nNo test images found."
        )

        sys.exit()

    print(
        f"\nFound {len(test_images)} test images."
    )

    print(
        "Testing first 5 images...\n"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for image_path in test_images[:5]:

        print("-" * 70)

        filename = os.path.basename(
            image_path
        )

        print(
            f"Image: {filename}"
        )

        try:

            result = pipeline.predict(
                image_path
            )

            print(
                f"Object detected: "
                f"{result['object_detected']}"
            )

            print(
                f"Prediction: "
                f"{result['class_name']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2f}%"
            )

            base_name = os.path.splitext(
                filename
            )[0]

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_boundary.jpg"
                ),
                result["boundary"]
            )

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_overlay.jpg"
                ),
                result["overlay"]
            )

            if result["crop"] is not None:

                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        f"{base_name}_crop.jpg"
                    ),
                    result["crop"]
                )

            print(
                "✓ Results saved"
            )

        except Exception as e:

            print(
                f"❌ Error: {e}"
            )

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)

    print(
        f"Results folder:\n"
        f"{output_dir}"
    )