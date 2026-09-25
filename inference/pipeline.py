# ============================================================
# CAMOUFLAGE BREAKER - FINAL INFERENCE PIPELINE
# SINet-V2 + ResNet50
# ============================================================

import os
import sys
import json
import urllib.request
import urllib.parse
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# SINET-V2 IMPORT
# ============================================================

SINET_SOURCE = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

sys.path.insert(0, SINET_SOURCE)

from lib.Network_Res2Net_GRA_NCD import Network


# ============================================================
# DEPLOYMENT MODEL DOWNLOAD
# ============================================================

def ensure_model_file(path, env_name):
    if os.path.exists(path):
        return

    default_urls = {
        "CAMOUFLAGE_SEGMENTATION_MODEL_URL":
            "https://github.com/Dhanush-S-007/Camouflage_Breaker/releases/download/v1.0-models/sinetv2_cod10k_40epoch_best.pth",
        "CAMOUFLAGE_CLASSIFIER_MODEL_URL":
            "https://github.com/Dhanush-S-007/Camouflage_Breaker/releases/download/v1.0-models/classifier_best.pth",
    }

    url = os.getenv(
        env_name,
        default_urls.get(env_name, "")
    ).strip()

    if not url:
        raise FileNotFoundError(
            f"Required model file not found: {path}\n"
            f"Set {env_name} to a direct downloadable model URL."
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"Downloading model for {env_name}...")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{env_name} must be an http/https URL."
        )

    fd, temp_path = tempfile.mkstemp(
        prefix="model_",
        suffix=".download",
        dir=os.path.dirname(path)
    )
    os.close(fd)

    try:
        urllib.request.urlretrieve(url, temp_path)
        os.replace(temp_path, path)
        print(f"Model ready: {path}")
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# ============================================================
# RESNET50 CLASSIFIER
# ============================================================

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


# ============================================================
# CAMOUFLAGE BREAKER PIPELINE
# ============================================================

class CamouflageBreakerPipeline:

    def __init__(
        self,
        seg_model_path=None,
        classifier_model_path=None,
        class_mapping_path=None
    ):

        # ----------------------------------------------------
        # MODEL PATHS
        # ----------------------------------------------------

        if seg_model_path is None:

            seg_model_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "sinetv2",
                "sinetv2_cod10k_40epoch_best.pth"
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
        # DEVICE
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
        # LOAD SINET-V2
        # ----------------------------------------------------

        print(
            "\nLoading SINet-V2 segmentation model..."
        )

        ensure_model_file(
            seg_model_path,
            "CAMOUFLAGE_SEGMENTATION_MODEL_URL"
        )

        self.seg_model = Network(
            channel=32,
            imagenet_pretrained=False
        )

        checkpoint = torch.load(
            seg_model_path,
            map_location=self.device
        )

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

        self.seg_model.load_state_dict(
            state_dict,
            strict=True
        )

        self.seg_model.to(
            self.device
        )

        self.seg_model.eval()

        print(
            "✓ SINet-V2 loaded"
        )

        # ----------------------------------------------------
        # LOAD CLASSIFIER
        # ----------------------------------------------------

        print(
            "\nLoading trained ResNet50 classifier..."
        )

        ensure_model_file(
            classifier_model_path,
            "CAMOUFLAGE_CLASSIFIER_MODEL_URL"
        )

        classifier_checkpoint = torch.load(
            classifier_model_path,
            map_location=self.device
        )

        if (
            isinstance(
                classifier_checkpoint,
                dict
            )
            and
            "num_classes"
            in classifier_checkpoint
        ):

            self.num_classes = int(
                classifier_checkpoint[
                    "num_classes"
                ]
            )

        else:

            self.num_classes = 69

        self.classifier = ResNet50Classifier(
            num_classes=self.num_classes
        )

        if (
            isinstance(
                classifier_checkpoint,
                dict
            )
            and
            "model_state_dict"
            in classifier_checkpoint
        ):

            classifier_state = (
                classifier_checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            classifier_state = (
                classifier_checkpoint
            )

        self.classifier.load_state_dict(
            classifier_state
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
        # CLASS MAPPING
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
        # CLASSIFIER TRANSFORM
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
    # PREPROCESS FOR SINET-V2
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
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        image_tensor = (
            image_tensor - mean
        ) / std

        image_tensor = (
            image_tensor.unsqueeze(0)
        )

        return image_tensor.to(
            self.device
        )

    # ========================================================
    # SINET-V2 SEGMENTATION
    # ========================================================

    def get_mask(
        self,
        image_tensor,
        threshold=0.5
    ):

        with torch.no_grad():

            outputs = self.seg_model(
                image_tensor
            )

            # IMPORTANT:
            # SINet-V2 returns multiple outputs.
            # The final prediction is outputs[-1].

            if isinstance(
                outputs,
                (tuple, list)
            ):

                prediction = outputs[-1]

            else:

                prediction = outputs

            probability = torch.sigmoid(
                prediction
            )

            probability = (
                probability.squeeze()
                .detach()
                .cpu()
                .numpy()
            )

        binary_mask = (
            probability >= threshold
        ).astype(
            np.uint8
        )

        return binary_mask, probability

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
    # OBJECT GATE
    # ========================================================

    def calculate_object_gate(
        self,
        mask,
        probability
    ):

        height, width = mask.shape

        total_pixels = (
            height * width
        )

        object_pixels = np.sum(
            mask > 0
        )

        area_ratio = (
            object_pixels /
            max(total_pixels, 1)
        )

        # ----------------------------------------------------
        # Largest connected component
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        largest_component_ratio = 0.0

        if num_labels > 1:

            component_areas = (
                stats[1:, cv2.CC_STAT_AREA]
            )

            largest_area = (
                np.max(component_areas)
            )

            largest_component_ratio = (
                largest_area /
                max(total_pixels, 1)
            )

        # ----------------------------------------------------
        # Probability statistics
        # ----------------------------------------------------

        if object_pixels > 0:

            object_probability = (
                probability[mask > 0]
            )

            mean_probability = float(
                np.mean(
                    object_probability
                )
            )

            max_probability = float(
                np.max(
                    object_probability
                )
            )

        else:

            mean_probability = 0.0
            max_probability = 0.0

        # ----------------------------------------------------
        # Detection decision
        # ----------------------------------------------------

        object_detected = (

            area_ratio >= 0.002

            and

            largest_component_ratio >= 0.001

            and

            mean_probability >= 0.60

            and

            max_probability >= 0.70
        )

        return {

            "area_ratio":
                float(area_ratio),

            "largest_component_ratio":
                float(
                    largest_component_ratio
                ),

            "mean_probability":
                float(
                    mean_probability
                ),

            "max_probability":
                float(
                    max_probability
                ),

            "object_detected":
                bool(
                    object_detected
                )
        }

    # ========================================================
    # ROYAL ORANGE + RED BOUNDARY
    # ========================================================

    def draw_boundary(
        self,
        image,
        mask,
        thickness=4
    ):

        result = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area > 20:

                # BGR
                # Bright orange-red

                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    (0, 70, 255),
                    thickness,
                    cv2.LINE_AA
                )

        return result

    # ========================================================
    # ORANGE + RED OVERLAY
    # ========================================================

    def create_overlay(
        self,
        image,
        mask,
        alpha=0.42
    ):

        result = image.copy()

        # ----------------------------------------------------
        # Create soft orange-red mask
        # ----------------------------------------------------

        colored_mask = np.zeros_like(
            image
        )

        # Main color:
        # BGR = (0, 105, 255)
        # This is strong orange.

        colored_mask[
            mask > 0
        ] = (
            0,
            105,
            255
        )

        # ----------------------------------------------------
        # Orange overlay
        # ----------------------------------------------------

        result = cv2.addWeighted(
            image,
            1.0 - alpha,
            colored_mask,
            alpha,
            0
        )

        # ----------------------------------------------------
        # Add red-orange edge
        # ----------------------------------------------------

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
                    (0, 45, 255),
                    3,
                    cv2.LINE_AA
                )

        return result

    # ========================================================
    # CREATE SOFT GRADIENT OVERLAY
    # ========================================================

    def create_heat_overlay(
        self,
        image,
        mask,
        probability
    ):

        result = image.copy()

        # Normalize probability

        probability_normalized = np.clip(
            probability,
            0.0,
            1.0
        )

        # Convert probability to original size

        probability_resized = cv2.resize(
            probability_normalized,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Orange → Red intensity
        # ----------------------------------------------------

        heat = np.zeros_like(
            image,
            dtype=np.uint8
        )

        # Blue channel

        heat[:, :, 0] = 0

        # Green decreases as probability rises

        heat[:, :, 1] = (
            140 *
            (
                1.0 -
                probability_resized
            )
        ).astype(
            np.uint8
        )

        # Red increases with probability

        heat[:, :, 2] = (
            120 +
            135 *
            probability_resized
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        # Only show heat inside mask

        mask_bool = (
            mask > 0
        )

        heat_only = np.zeros_like(
            image
        )

        heat_only[
            mask_bool
        ] = heat[
            mask_bool
        ]

        # ----------------------------------------------------
        # Blend
        # ----------------------------------------------------

        result = cv2.addWeighted(
            image,
            0.58,
            heat_only,
            0.42,
            0
        )

        # ----------------------------------------------------
        # Red-orange boundary
        # ----------------------------------------------------

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
                    (0, 45, 255),
                    3,
                    cv2.LINE_AA
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

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.classifier(
                image_tensor
            )

            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1
                )
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

        class_name = (
            self.idx_to_class.get(
                str(predicted_index),
                f"Class_{predicted_index + 1}"
            )
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

        if image is None:

            raise ValueError(
                "Input image is None"
            )

        # ----------------------------------------------------
        # Ensure BGR uint8
        # ----------------------------------------------------

        if image.dtype != np.uint8:

            image = np.clip(
                image,
                0,
                255
            ).astype(
                np.uint8
            )

        original = image.copy()

        # ----------------------------------------------------
        # SINET-V2
        # ----------------------------------------------------

        image_tensor = (
            self.preprocess_image(
                image
            )
        )

        mask_small, probability_small = (
            self.get_mask(
                image_tensor,
                threshold
            )
        )

        # ----------------------------------------------------
        # Resize mask
        # ----------------------------------------------------

        mask = self.resize_mask(
            mask_small,
            image
        )

        # ----------------------------------------------------
        # Resize probability
        # ----------------------------------------------------

        probability = cv2.resize(
            probability_small,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # OBJECT GATE
        # ----------------------------------------------------

        gate = self.calculate_object_gate(
            mask,
            probability
        )

        print("\nObject gate:")
        print(
            f"Area ratio       : "
            f"{gate['area_ratio']:.4f}"
        )
        print(
            f"Largest component: "
            f"{gate['largest_component_ratio']:.4f}"
        )
        print(
            f"Mean probability : "
            f"{gate['mean_probability']:.4f}"
        )
        print(
            f"Max probability  : "
            f"{gate['max_probability']:.4f}"
        )
        print(
            f"Detected         : "
            f"{gate['object_detected']}"
        )

        # ----------------------------------------------------
        # NO OBJECT
        # ----------------------------------------------------

        if not gate["object_detected"]:

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
                    False,

                "gate":
                    gate
            }

        # ----------------------------------------------------
        # BOUNDARY
        # ----------------------------------------------------

        boundary = self.draw_boundary(
            original,
            mask
        )

        # ----------------------------------------------------
        # ORANGE-RED OVERLAY
        # ----------------------------------------------------

        overlay = self.create_overlay(
            original,
            mask
        )

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop = self.crop_object(
            original,
            mask
        )

        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

        (
            class_name,
            confidence,
            predicted_index
        ) = self.classify_object(
            crop
        )

        print(
            f"Prediction       : "
            f"{class_name}"
        )

        print(
            f"Confidence       : "
            f"{confidence:.2f}%"
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

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
                True,

            "gate":
                gate
        }


# ============================================================
# TEST PIPELINE
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
        f"\nFound "
        f"{len(test_images)} "
        f"test images."
    )

    print(
        "Testing first 5 images...\n"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "pipeline_test"
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

            base_name = (
                os.path.splitext(
                    filename
                )[0]
            )

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_original.jpg"
                ),
                result["original"]
            )

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_mask.png"
                ),
                result["mask"] * 255
            )

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

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Results folder:\n"
        f"{output_dir}"
    )
# ============================================================
# API PREDICTION FUNCTION
# Used by ai_service/main.py
# ============================================================

_pipeline = None


def get_pipeline():
    global _pipeline

    if _pipeline is None:
        _pipeline = CamouflageBreakerPipeline()

    return _pipeline


def predict(image):
    """
    Main prediction function used by FastAPI.
    """

    return get_pipeline().predict(image)