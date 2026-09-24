import streamlit as st
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import cv2


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import predict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Camouflage Breaker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(24, 185, 105, 0.08),
                transparent 25%
            ),
            #06100c;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* =========================
       TEXT
       ========================= */

    p {
        color: #a9b8b1 !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
    }

    h1 {
        color: #f7fbf9 !important;
        font-size: 50px !important;
        font-weight: 850 !important;
        letter-spacing: -2px !important;
        line-height: 1.05 !important;
    }

    h2 {
        color: #f0f6f3 !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #e7f0ec !important;
        font-size: 20px !important;
        font-weight: 750 !important;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #040907;
        border-right: 1px solid #17241f;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 27px !important;
    }

    section[data-testid="stSidebar"] p {
        font-size: 13px !important;
    }

    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #182820 !important;
        margin: 1.4rem 0 !important;
    }

    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: #0a1510;
        border: 1px dashed #33734f;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #48dc8b;
    }

    /* =========================
       BUTTON
       ========================= */

    .stButton > button {
        min-height: 55px;
        border-radius: 12px;
        border: 1px solid #36dc80;
        background: linear-gradient(
            135deg,
            #15b968,
            #29d579
        );
        color: #021109;
        font-size: 16px;
        font-weight: 850;
        box-shadow: 0 8px 25px rgba(25, 210, 115, 0.12);
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #21ca72,
            #39e48b
        );
        border-color: #70f5aa;
    }

    /* =========================
       DOWNLOAD
       ========================= */

    .stDownloadButton > button {
        min-height: 52px;
        border-radius: 11px;
        border: 1px solid #287b50;
        background: #0b2117;
        color: #52e596;
        font-weight: 800;
    }

    .stDownloadButton > button:hover {
        background: #102f20;
        border-color: #42dc88;
    }

    /* =========================
       METRICS
       ========================= */

    [data-testid="stMetric"] {
        background: #0a1611;
        border: 1px solid #1a3025;
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #80948a !important;
        font-size: 13px !important;
    }

    [data-testid="stMetricValue"] {
        color: #52e596 !important;
        font-size: 26px !important;
        font-weight: 850 !important;
    }

    /* =========================
       IMAGE CONTAINERS
       ========================= */

    [data-testid="stImage"] {
        background: #08120e;
        border: 1px solid #1b3127;
        border-radius: 13px;
        padding: 5px;
    }

    [data-testid="stImage"] img {
        border-radius: 9px;
    }

    /* =========================
       EXPANDER
       ========================= */

    [data-testid="stExpander"] {
        background: #09140f;
        border: 1px solid #1a2e24;
        border-radius: 13px;
    }

    /* =========================
       HIDE STREAMLIT BRANDING
       ========================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def bgr_to_rgb(image):
    """
    Convert OpenCV BGR image to RGB for Streamlit.
    """
    if image is None:
        return None

    image = np.asarray(image)

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    return image


def prepare_display_image(
    image,
    max_width=560,
    max_height=420
):
    """
    Resize only for website display.
    Original model output is not modified.
    """

    if image is None:
        return None

    image = np.asarray(image)

    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale < 1.0:

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


def display_image(
    image,
    max_width=560,
    max_height=420,
    caption=None
):
    """
    Correct color + controlled display size.
    """

    rgb = bgr_to_rgb(
        image
    )

    rgb = prepare_display_image(
        rgb,
        max_width=max_width,
        max_height=max_height
    )

    st.image(
        rgb,
        caption=caption,
        use_container_width=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎯 Camouflage")
    st.title("Breaker")

    st.caption(
        "AI-powered camouflaged object detection"
    )

    st.divider()

    st.subheader("SYSTEM")

    st.success(
        "● SYSTEM READY"
    )

    st.divider()

    st.subheader("AI MODELS")

    st.write("**SINet-V2**")

    st.caption(
        "Camouflaged object segmentation"
    )

    st.write("**ResNet50**")

    st.caption(
        "Object classification · 69 classes"
    )

    st.divider()

    st.subheader("MODEL PERFORMANCE")

    st.metric(
        "Mean IoU",
        "71.91%"
    )

    st.metric(
        "Mean Dice",
        "76.57%"
    )

    st.divider()

    st.subheader("DETECTION FLOW")

    st.write(
        "📤 Upload\n\n"
        "🎯 Locate\n\n"
        "🟢 Highlight\n\n"
        "✂️ Isolate\n\n"
        "🧠 Identify"
    )

    st.divider()

    st.caption(
        "COD10K · Deep Learning · Computer Vision"
    )


# ============================================================
# HERO
# ============================================================

st.caption(
    "● AI COMPUTER VISION SYSTEM"
)

st.title(
    "See what is hidden."
)

st.write(
    "Find camouflaged objects, reveal their boundaries, "
    "isolate them from the scene, and identify what is hidden."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "Upload an image"
)

st.caption(
    "JPG · JPEG · PNG"
)

uploaded_file = st.file_uploader(
    "Choose image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.divider()

    st.subheader(
        "How Camouflage Breaker works"
    )

    st.caption(
        "A complete AI pipeline designed specifically for hidden objects."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "01",
            "UPLOAD"
        )

    with c2:
        st.metric(
            "02",
            "LOCATE"
        )

    with c3:
        st.metric(
            "03",
            "SEGMENT"
        )

    with c4:
        st.metric(
            "04",
            "IDENTIFY"
        )

    with c5:
        st.metric(
            "05",
            "EXPORT"
        )

    st.divider()

    st.subheader(
        "Two AI models. One objective."
    )

    a, b = st.columns(2)

    with a:

        st.info(
            "**SINet-V2 — WHERE?**\n\n"
            "Finds the pixels belonging to the camouflaged "
            "object and creates its segmentation mask."
        )

    with b:

        st.info(
            "**ResNet50 — WHAT?**\n\n"
            "Analyzes the isolated object and predicts "
            "its category from 69 trained classes."
        )


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    width, height = image.size

    st.divider()

    # ========================================================
    # INPUT PREVIEW
    # ========================================================

    st.subheader(
        "Input image"
    )

    input_left, input_right = st.columns(
        [1.25, 0.75]
    )

    with input_left:

        st.image(
            image,
            width=560
        )

    with input_right:

        st.metric(
            "Resolution",
            f"{width} × {height}"
        )

        st.metric(
            "Format",
            uploaded_file.type.replace(
                "image/",
                ""
            ).upper()
        )

        st.metric(
            "Segmentation",
            "SINet-V2"
        )

        st.caption(
            "The image will be analysed using "
            "the trained camouflage segmentation model."
        )

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🚀 ANALYZE CAMOUFLAGE",
        use_container_width=True
    )

    if analyze:

        try:

            # =================================================
            # MODEL
            # =================================================

            with st.spinner(
                "Analysing hidden object..."
            ):

                image_array = np.array(
                    image
                )

                result = predict(
                    image_array
                )

            st.success(
                "Analysis completed."
            )

            # =================================================
            # RESULTS
            # =================================================

            original = result[
                "original"
            ]

            boundary = result[
                "boundary"
            ]

            overlay = result[
                "overlay"
            ]

            mask = result[
                "mask"
            ]

            crop = result[
                "crop"
            ]

            class_name = result[
                "class_name"
            ]

            confidence = float(
                result[
                    "confidence"
                ]
            )

            # =================================================
            # MAIN DETECTION
            # =================================================

            st.subheader(
                "Detection result"
            )

            st.caption(
                "Compare the original scene with the detected camouflage."
            )

            result_left, result_right = st.columns(
                2
            )

            with result_left:

                st.write(
                    "### Original"
                )

                display_image(
                    original,
                    max_width=560,
                    max_height=390,
                    caption="Original uploaded image"
                )

            with result_right:

                st.write(
                    "### Detected"
                )

                display_image(
                    overlay,
                    max_width=560,
                    max_height=390,
                    caption="Camouflaged object highlighted"
                )

            st.divider()

            # =================================================
            # AI PREDICTION
            # =================================================

            st.subheader(
                "What did the AI find?"
            )

            prediction_left, prediction_right = st.columns(
                2
            )

            with prediction_left:

                st.metric(
                    "Predicted object",
                    class_name
                )

                st.caption(
                    "ResNet50 classification result"
                )

            with prediction_right:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            confidence / 100.0,
                            0.0
                        ),
                        1.0
                    )
                )

            st.divider()

            # =================================================
            # ANALYSIS OUTPUTS
            # =================================================

            st.subheader(
                "Detection outputs"
            )

            st.caption(
                "Every important output generated by the pipeline."
            )

            # -------------------------------------------------
            # ROW 1
            # -------------------------------------------------

            out1, out2, out3 = st.columns(
                3
            )

            with out1:

                st.write(
                    "### 🎯 Boundary"
                )

                display_image(
                    boundary,
                    max_width=420,
                    max_height=300,
                    caption="Object boundary"
                )

            with out2:

                st.write(
                    "### 🟢 Overlay"
                )

                display_image(
                    overlay,
                    max_width=420,
                    max_height=300,
                    caption="Detection overlay"
                )

            with out3:

                st.write(
                    "### ⚪ Mask"
                )

                display_image(
                    mask,
                    max_width=420,
                    max_height=300,
                    caption="Segmentation mask"
                )

            # -------------------------------------------------
            # CROP
            # -------------------------------------------------

            st.write("")

            crop_left, crop_right = st.columns(
                [0.9, 1.1]
            )

            with crop_left:

                st.write(
                    "### ✂️ Isolated object"
                )

                if crop is not None:

                    display_image(
                        crop,
                        max_width=430,
                        max_height=320,
                        caption=f"Detected: {class_name}"
                    )

                else:

                    st.warning(
                        "Object crop could not be generated."
                    )

            with crop_right:

                st.write(
                    "### Identification"
                )

                st.metric(
                    "Object",
                    class_name
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.write(
                    "The isolated object is passed to "
                    "the trained ResNet50 classifier."
                )

            st.divider()

            # =================================================
            # AI PIPELINE
            # =================================================

            st.subheader(
                "AI pipeline"
            )

            p1, p2, p3, p4 = st.columns(
                4
            )

            with p1:

                st.write(
                    "### 01"
                )

                st.write(
                    "**Locate**"
                )

                st.caption(
                    "SINet-V2 searches for the hidden object."
                )

            with p2:

                st.write(
                    "### 02"
                )

                st.write(
                    "**Segment**"
                )

                st.caption(
                    "The object is converted into a pixel mask."
                )

            with p3:

                st.write(
                    "### 03"
                )

                st.write(
                    "**Isolate**"
                )

                st.caption(
                    "The detected region is cropped from the scene."
                )

            with p4:

                st.write(
                    "### 04"
                )

                st.write(
                    "**Identify**"
                )

                st.caption(
                    "ResNet50 predicts the object category."
                )

            st.divider()

            # =================================================
            # MODEL INFORMATION
            # =================================================

            with st.expander(
                "🧠 Model information"
            ):

                model_left, model_right = st.columns(
                    2
                )

                with model_left:

                    st.write(
                        "### SINet-V2"
                    )

                    st.write(
                        "**Role:** Camouflaged object segmentation"
                    )

                    st.write(
                        "**Input:** 352 × 352"
                    )

                    st.write(
                        "**Dataset:** COD10K"
                    )

                    st.write(
                        "**Mean IoU:** 71.91%"
                    )

                    st.write(
                        "**Mean Dice:** 76.57%"
                    )

                with model_right:

                    st.write(
                        "### ResNet50"
                    )

                    st.write(
                        "**Role:** Object classification"
                    )

                    st.write(
                        "**Input:** 224 × 224"
                    )

                    st.write(
                        "**Classes:** 69"
                    )

                    st.write(
                        "**Dataset:** COD10K"
                    )

            # =================================================
            # DOWNLOAD
            # =================================================

            st.subheader(
                "Export"
            )

            st.caption(
                "Download the detected image."
            )

            # Convert OpenCV BGR → RGB before saving.
            overlay_rgb = bgr_to_rgb(
                overlay
            )

            output_image = Image.fromarray(
                overlay_rgb
            )

            image_bytes = io.BytesIO()

            output_image.save(
                image_bytes,
                format="JPEG",
                quality=95
            )

            image_bytes.seek(0)

            st.download_button(
                label="⬇️ DOWNLOAD DETECTED IMAGE",
                data=image_bytes,
                file_name="camouflage_breaker_result.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                "Image analysis failed."
            )

            st.exception(
                error
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎯 Camouflage Breaker  •  SINet-V2  •  ResNet50  •  COD10K"
)