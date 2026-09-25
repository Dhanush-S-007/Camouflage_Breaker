import io
import zipfile

import cv2
import numpy as np
import streamlit as st

from inference.pipeline import get_pipeline


st.set_page_config(
    page_title="Camouflage Breaker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 80% 10%, rgba(255, 92, 0, 0.08), transparent 28%),
            radial-gradient(circle at 15% 20%, rgba(255, 196, 92, 0.06), transparent 25%),
            #07090d;
        color: #f5f1e8;
        font-family: 'Manrope', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.04em;
    }

    .mono {
        font-family: 'DM Mono', monospace;
        color: #d7b46a;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero {
        padding: 1.5rem 0 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: clamp(3rem, 7vw, 6.2rem);
        line-height: 0.9;
        margin: 0.35rem 0 1rem;
    }

    .hero p {
        max-width: 720px;
        color: #a9adb5;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .status {
        display: inline-flex;
        gap: 0.55rem;
        align-items: center;
        padding: 0.45rem 0.7rem;
        border: 1px solid rgba(255,196,92,0.25);
        border-radius: 999px;
        background: rgba(255,196,92,0.05);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #e6c77e;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #e6c77e;
        box-shadow: 0 0 12px rgba(230,199,126,0.8);
    }

    .result-card {
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: rgba(255,255,255,0.025);
        margin: 0.5rem 0 1rem;
    }

    .result-label {
        color: #8e949f;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .result-value {
        color: #f7efe1;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    .confidence {
        color: #ff9a3c;
        font-family: 'DM Mono', monospace;
        font-size: 1rem;
        margin-top: 0.15rem;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,170,70,0.35);
        background: linear-gradient(135deg, #d97818, #a94316);
        color: white;
        font-weight: 800;
        min-height: 2.8rem;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #ffc45c;
        color: white;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255,196,92,0.32);
        border-radius: 16px;
        padding: 0.5rem;
        background: rgba(255,255,255,0.018);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="status"><span class="dot"></span> AI SYSTEM ONLINE</div>
        <div class="mono" style="margin-top:1.3rem;">SINet-V2 · ResNet50 · COD10K</div>
        <h1>SEE WHAT<br>HIDES.</h1>
        <p>
            Upload an image and Camouflage Breaker will segment the concealed
            object, highlight its boundary, extract the object, and predict
            its class with a confidence score.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "UPLOAD IMAGE",
    type=["jpg", "jpeg", "png"],
    help="Maximum recommended size: 10 MB.",
)

if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Image is larger than 10 MB. Please choose a smaller image.")
        st.stop()

    data = uploaded.getvalue()

    file_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read this image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption=uploaded.name,
        use_container_width=True,
    )

    if st.button("DETECT CAMOUFLAGE", type="primary", use_container_width=True):
        try:
            with st.spinner("Running SINet-V2 segmentation and ResNet50 classification..."):
                pipeline = get_pipeline()
                result = pipeline.predict(image)

            if not result["object_detected"]:
                st.warning(
                    "No sufficiently strong camouflaged object was detected in this image."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
                        caption="Original",
                        use_container_width=True,
                    )
                with c2:
                    st.image(
                        result["mask"] * 255,
                        caption="Segmentation Mask",
                        use_container_width=True,
                    )

            else:
                st.success("Camouflaged object detected.")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        '<div class="result-card"><div class="result-label">Prediction</div>'
                        f'<div class="result-value">{result["class_name"]}</div></div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(
                        '<div class="result-card"><div class="result-label">Confidence</div>'
                        f'<div class="result-value">{result["confidence"]:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )

                with c3:
                    st.markdown(
                        '<div class="result-card"><div class="result-label">Object Area</div>'
                        f'<div class="result-value">{result["gate"]["area_ratio"] * 100:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("### ANALYSIS OUTPUT")

                a, b = st.columns(2)
                with a:
                    st.image(
                        cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
                        caption="Original",
                        use_container_width=True,
                    )
                with b:
                    st.image(
                        cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB),
                        caption="Camouflage Overlay",
                        use_container_width=True,
                    )

                c, d = st.columns(2)
                with c:
                    st.image(
                        cv2.cvtColor(result["boundary"], cv2.COLOR_BGR2RGB),
                        caption="Object Boundary",
                        use_container_width=True,
                    )
                with d:
                    if result["crop"] is not None:
                        st.image(
                            cv2.cvtColor(result["crop"], cv2.COLOR_BGR2RGB),
                            caption="Detected Object Crop",
                            use_container_width=True,
                        )

                files = {
                    "original.jpg": result["original"],
                    "mask.png": result["mask"] * 255,
                    "boundary.jpg": result["boundary"],
                    "overlay.jpg": result["overlay"],
                }

                if result["crop"] is not None:
                    files["crop.jpg"] = result["crop"]

                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    zip_buffer,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as archive:
                    for filename, array in files.items():
                        ext = filename.rsplit(".", 1)[-1]
                        success, encoded = cv2.imencode(
                            "." + ext,
                            array,
                        )

                        if success:
                            archive.writestr(
                                filename,
                                encoded.tobytes(),
                            )

                zip_buffer.seek(0)

                st.download_button(
                    "DOWNLOAD RESULT PACKAGE",
                    data=zip_buffer.getvalue(),
                    file_name="camouflage_breaker_results.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

else:
    st.info("Upload a JPG, JPEG, or PNG image to begin.")
