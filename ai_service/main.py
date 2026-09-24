# ============================================================
# Camouflage Breaker - FastAPI AI Service
# ============================================================

import base64
import io
import os
import sys

import numpy as np

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT AI PIPELINE
# ============================================================

from inference.pipeline import predict


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Camouflage Breaker AI Service",
    description="SINet-V2 + Object Gate + ResNet50",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image,
    extension=".jpg"
):

    if image is None:
        return None

    if extension == ".png":

        success, encoded = (
            __import__("cv2").imencode(
                ".png",
                image
            )
        )

    else:

        success, encoded = (
            __import__("cv2").imencode(
                ".jpg",
                image
            )
        )

    if not success:
        return None

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")


# ============================================================
# RESULT IMAGE HELPER
# ============================================================

def get_result_image(
    result,
    key
):

    image = result.get(
        key
    )

    if image is None:
        return None

    return image_to_base64(
        image,
        ".jpg"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "ok",
        "service": "Camouflage Breaker AI Service",
        "pipeline": "SINet-V2 + Object Gate + ResNet50"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict"
)
async def predict_image(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if file is None:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "No file uploaded"
                }
            )

        if not file.filename:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Filename is missing"
                }
            )

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp"
        }

        if (
            file.content_type
            and file.content_type
            not in allowed_types
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Unsupported image format",
                    "supported_formats": [
                        "JPEG",
                        "PNG",
                        "WEBP",
                        "BMP"
                    ]
                }
            )

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Uploaded file is empty"
                }
            )

        # ----------------------------------------------------
        # PIL → RGB → NumPy
        # ----------------------------------------------------

        pil_image = Image.open(
            io.BytesIO(contents)
        ).convert(
            "RGB"
        )

        image_array = np.array(
            pil_image
        )

        # ----------------------------------------------------
        # Run AI pipeline
        # ----------------------------------------------------

        result = predict(
            image_array
        )

        # ----------------------------------------------------
        # Extract AI result
        # ----------------------------------------------------

        object_detected = bool(
            result.get(
                "object_detected",
                False
            )
        )

        if object_detected:

            class_name = result.get(
                "class_name",
                "Unknown"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            predicted_index = result.get(
                "predicted_index"
            )

        else:

            class_name = (
                "No object detected"
            )

            confidence = 0.0

            predicted_index = None

        # ----------------------------------------------------
        # Gate information
        # ----------------------------------------------------

        gate = result.get(
            "gate",
            {}
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        response = {

            "success": True,

            "object_detected":
                object_detected,

            "filename":
                file.filename,

            "class_name":
                class_name,

            "confidence":
                confidence,

            "predicted_index":
                predicted_index,

            "gate":
                gate,

            "original":
                get_result_image(
                    result,
                    "original"
                ),

            "mask":
                get_result_image(
                    result,
                    "mask"
                ),

            "boundary":
                get_result_image(
                    result,
                    "boundary"
                ),

            "overlay":
                get_result_image(
                    result,
                    "overlay"
                ),

            "crop":
                get_result_image(
                    result,
                    "crop"
                )
        }

        return JSONResponse(
            status_code=200,
            content=response
        )

    except Exception as e:

        print(
            "\nAI SERVICE ERROR:"
        )

        print(
            str(e)
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "AI prediction failed",
                "message": str(e)
            }
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "Camouflage Breaker AI Service",

        "version":
            "1.0.0",

        "pipeline":
            "SINet-V2 + Object Gate + ResNet50",

        "endpoints": {

            "health":
                "/health",

            "predict":
                "/predict"

        }
    }