from pathlib import Path
import sys

from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Test"
    / "Image"
)


# ============================================================
# CHECK TEST DATASET
# ============================================================

if not TEST_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Test image folder not found:\n{TEST_IMAGE_DIR}"
    )


image_files = sorted(
    [
        p
        for p in TEST_IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]
)


if not image_files:
    raise RuntimeError(
        "No test images found."
    )


# ============================================================
# IMPORT PIPELINE
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from inference.pipeline import predict


# ============================================================
# SELECT FIRST TEST IMAGE
# ============================================================

image_path = image_files[0]

print("=" * 70)
print("CAMOUFLAGE BREAKER — PIPELINE TEST")
print("=" * 70)

print("Test image:")
print(image_path)

print()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    image_path
).convert("RGB")

print(
    "Original image size:",
    image.size
)

print()


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

print("Running complete pipeline...")
print()

result = predict(
    image
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("PIPELINE RESULT")
print("=" * 70)

print(
    "Predicted object :",
    result["class_name"]
)

print(
    "Confidence       :",
    f"{result['confidence'] * 100:.2f}%"
)

print(
    "Mask size        :",
    result["mask"].size
)

print(
    "Crop size        :",
    result["crop"].size
)


# ============================================================
# SAVE TEST OUTPUTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


original_output = (
    OUTPUT_DIR
    / "original.jpg"
)

mask_output = (
    OUTPUT_DIR
    / "mask.png"
)

boundary_output = (
    OUTPUT_DIR
    / "boundary.jpg"
)

overlay_output = (
    OUTPUT_DIR
    / "overlay.jpg"
)

crop_output = (
    OUTPUT_DIR
    / "crop.jpg"
)


result["original"].save(
    original_output
)

result["mask"].save(
    mask_output
)

result["boundary"].save(
    boundary_output
)

result["overlay"].save(
    overlay_output
)

result["crop"].save(
    crop_output
)


# ============================================================
# FINAL CHECK
# ============================================================

output_files = [
    original_output,
    mask_output,
    boundary_output,
    overlay_output,
    crop_output,
]


print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

for output_file in output_files:

    if output_file.exists():

        print(
            "OK:",
            output_file
        )

    else:

        raise RuntimeError(
            f"Expected output was not created:\n"
            f"{output_file}"
        )


print()
print("=" * 70)
print("PIPELINE TEST SUCCESSFUL")
print("=" * 70)