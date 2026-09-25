/* ============================================================
   CAMOUFLAGE BREAKER
   FRONTEND APPLICATION
============================================================ */

const JAVA_API = "/api";

let selectedFile = null;
let currentResult = null;


/* ============================================================
   ELEMENTS
============================================================ */

const fileInput =
    document.getElementById("fileInput");

const uploadCard =
    document.getElementById("uploadCard");

const uploadContent =
    document.getElementById("uploadContent");

const previewArea =
    document.getElementById("previewArea");

const previewImage =
    document.getElementById("previewImage");

const previewFilename =
    document.getElementById("previewFilename");

const removeButton =
    document.getElementById("removeButton");

const detectButton =
    document.getElementById("detectButton");

const fileStatus =
    document.getElementById("fileStatus");

const processingPanel =
    document.getElementById("processingPanel");

const progressFill =
    document.getElementById("progressFill");

const resultsSection =
    document.getElementById("resultsSection");

const errorPanel =
    document.getElementById("errorPanel");

const errorTitle =
    document.getElementById("errorTitle");

const errorMessage =
    document.getElementById("errorMessage");

const predictionName =
    document.getElementById("predictionName");

const predictionTitle =
    document.getElementById("predictionTitle");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceValueLarge =
    document.getElementById(
        "confidenceValueLarge"
    );

const confidenceFill =
    document.getElementById("confidenceFill");

const resultBadge =
    document.getElementById("resultBadge");

const resultStatusDot =
    document.getElementById(
        "resultStatusDot"
    );

const originalResult =
    document.getElementById(
        "originalResult"
    );

const overlayResult =
    document.getElementById(
        "overlayResult"
    );

const boundaryResult =
    document.getElementById(
        "boundaryResult"
    );

const cropResult =
    document.getElementById(
        "cropResult"
    );

const downloadButton =
    document.getElementById(
        "downloadButton"
    );

const newScanButton =
    document.getElementById(
        "newScanButton"
    );

const stepSegment =
    document.getElementById(
        "stepSegment"
    );

const stepPattern =
    document.getElementById(
        "stepPattern"
    );

const stepClassify =
    document.getElementById(
        "stepClassify"
    );


/* ============================================================
   FILE SELECT
============================================================ */

uploadCard.addEventListener(
    "click",
    () => {

        if (!selectedFile) {

            fileInput.click();

        }

    }
);


fileInput.addEventListener(
    "change",
    (event) => {

        const file =
            event.target.files[0];

        if (file) {

            handleFile(file);

        }

    }
);


/* ============================================================
   DRAG & DROP
============================================================ */

uploadCard.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        uploadCard.classList.add(
            "dragging"
        );

    }
);


uploadCard.addEventListener(
    "dragleave",
    () => {

        uploadCard.classList.remove(
            "dragging"
        );

    }
);


uploadCard.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        uploadCard.classList.remove(
            "dragging"
        );

        const file =
            event.dataTransfer.files[0];

        if (file) {

            handleFile(file);

        }

    }
);


/* ============================================================
   HANDLE FILE
============================================================ */

function handleFile(file) {

    hideError();

    const validTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    if (
        !validTypes.includes(
            file.type
        )
    ) {

        showError(
            "INVALID FILE",
            "Please upload a JPG, JPEG, or PNG image."
        );

        return;

    }


    const maxSize =
        10 * 1024 * 1024;

    if (
        file.size > maxSize
    ) {

        showError(
            "FILE TOO LARGE",
            "Maximum allowed image size is 10 MB."
        );

        return;

    }


    selectedFile = file;

    const reader =
        new FileReader();


    reader.onload = (event) => {

        previewImage.src =
            event.target.result;

        previewFilename.textContent =
            file.name;

        uploadContent.style.display =
            "none";

        previewArea.style.display =
            "block";

        detectButton.disabled =
            false;

        fileStatus.textContent =
            `${formatFileSize(file.size)} • READY FOR ANALYSIS`;

    };


    reader.readAsDataURL(file);

}


/* ============================================================
   FILE SIZE
============================================================ */

function formatFileSize(bytes) {

    if (
        bytes < 1024
    ) {

        return `${bytes} B`;

    }

    if (
        bytes < 1024 * 1024
    ) {

        return `${(
            bytes / 1024
        ).toFixed(1)} KB`;

    }

    return `${(
        bytes /
        (1024 * 1024)
    ).toFixed(2)} MB`;

}


/* ============================================================
   REMOVE FILE
============================================================ */

removeButton.addEventListener(
    "click",
    (event) => {

        event.stopPropagation();

        resetApplication();

    }
);


/* ============================================================
   DETECT
============================================================ */

detectButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            return;

        }

        hideError();

        resultsSection.style.display =
            "none";

        processingPanel.style.display =
            "block";

        detectButton.disabled =
            true;

        startProcessingAnimation();


        const formData =
            new FormData();

        formData.append(
            "file",
            selectedFile
        );


        try {

            const response =
                await fetch(
                    `${JAVA_API}/predict`,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            if (!response.ok) {

                throw new Error(
                    `Server returned ${response.status}`
                );

            }


            const data =
                await response.json();


            if (!data.success) {

                throw new Error(
                    data.message ||
                    data.error ||
                    "AI detection failed."
                );

            }


            currentResult = data;

            finishProcessingAnimation();

            displayResults(data);


        } catch (error) {

            processingPanel.style.display =
                "none";

            detectButton.disabled =
                false;

            showError(
                "DETECTION FAILED",
                error.message ||
                "Unable to connect to the AI service."
            );

        }

    }
);


/* ============================================================
   PROCESSING ANIMATION
============================================================ */

function startProcessingAnimation() {

    progressFill.style.width =
        "10%";

    stepSegment.classList.add(
        "active"
    );

    stepPattern.classList.remove(
        "active"
    );

    stepClassify.classList.remove(
        "active"
    );

    stepSegment.querySelector(
        ".step-state"
    ).textContent = "...";

    stepPattern.querySelector(
        ".step-state"
    ).textContent = "WAIT";

    stepClassify.querySelector(
        ".step-state"
    ).textContent = "WAIT";


    setTimeout(
        () => {

            progressFill.style.width =
                "45%";

            stepPattern.classList.add(
                "active"
            );

            stepSegment.querySelector(
                ".step-state"
            ).textContent = "DONE";

            stepPattern.querySelector(
                ".step-state"
            ).textContent = "...";

        },
        700
    );


    setTimeout(
        () => {

            progressFill.style.width =
                "72%";

            stepClassify.classList.add(
                "active"
            );

            stepPattern.querySelector(
                ".step-state"
            ).textContent = "DONE";

            stepClassify.querySelector(
                ".step-state"
            ).textContent = "...";

        },
        1400
    );

}


function finishProcessingAnimation() {

    progressFill.style.width =
        "100%";

    stepSegment.querySelector(
        ".step-state"
    ).textContent = "DONE";

    stepPattern.querySelector(
        ".step-state"
    ).textContent = "DONE";

    stepClassify.querySelector(
        ".step-state"
    ).textContent = "DONE";

}


/* ============================================================
   DISPLAY RESULTS
============================================================ */

function displayResults(data) {

    const detected =
        data.object_detected === true;


    const className =
        data.class_name ||
        "No object detected";


    const confidence =
        Number(
            data.confidence || 0
        );


    predictionName.textContent =
        className;

    predictionTitle.textContent =
        className;


    confidenceValue.textContent =
        `${confidence.toFixed(1)}%`;

    confidenceValueLarge.textContent =
        `${confidence.toFixed(1)}%`;

    confidenceFill.style.width =
        `${Math.min(
            confidence,
            100
        )}%`;


    if (detected) {

        resultBadge.textContent =
            "OBJECT DETECTED";

        resultStatusDot.style.background =
            "var(--orange)";

        resultStatusDot.style.boxShadow =
            "0 0 15px var(--orange)";

        if (data.crop) {

            cropResult.src =
                convertBase64(
                    data.crop
                );

        }

    } else {

        resultBadge.textContent =
            "NO OBJECT DETECTED";

        resultStatusDot.style.background =
            "var(--muted)";

        resultStatusDot.style.boxShadow =
            "none";

        predictionName.textContent =
            "No object detected";

        predictionTitle.textContent =
            "No object detected";

        confidenceValue.textContent =
            "0%";

        confidenceValueLarge.textContent =
            "0%";

        confidenceFill.style.width =
            "0%";

    }


    if (data.original) {

        originalResult.src =
            convertBase64(
                data.original
            );

    }


    if (data.overlay) {

        overlayResult.src =
            convertBase64(
                data.overlay
            );

    }


    if (data.boundary) {

        boundaryResult.src =
            convertBase64(
                data.boundary
            );

    }


    if (data.crop) {

        cropResult.src =
            convertBase64(
                data.crop
            );

    } else {

        cropResult.removeAttribute(
            "src"
        );

    }


    processingPanel.style.display =
        "none";

    resultsSection.style.display =
        "block";

    detectButton.disabled =
        false;


    setTimeout(
        () => {

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        100
    );

}


/* ============================================================
   BASE64 IMAGE
============================================================ */

function convertBase64(value) {

    if (!value) {

        return "";

    }

    if (
        value.startsWith(
            "data:image"
        )
    ) {

        return value;

    }

    return `data:image/jpeg;base64,${value}`;

}


/* ============================================================
   BASE64 → BLOB
============================================================ */

function base64ToBlob(
    base64,
    contentType = "image/jpeg"
) {

    if (!base64) {

        return null;

    }


    const cleanBase64 =
        base64.includes(",")
            ? base64.split(",")[1]
            : base64;


    const byteCharacters =
        atob(cleanBase64);

    const byteArrays = [];


    const sliceSize =
        1024;


    for (
        let offset = 0;
        offset < byteCharacters.length;
        offset += sliceSize
    ) {

        const slice =
            byteCharacters.slice(
                offset,
                offset + sliceSize
            );


        const byteNumbers =
            new Array(
                slice.length
            );


        for (
            let i = 0;
            i < slice.length;
            i++
        ) {

            byteNumbers[i] =
                slice.charCodeAt(i);

        }


        byteArrays.push(
            new Uint8Array(
                byteNumbers
            )
        );

    }


    return new Blob(
        byteArrays,
        {
            type: contentType
        }
    );

}


/* ============================================================
   CREATE TEXT FILE
============================================================ */

function createTextBlob(
    data
) {

    const detected =
        data.object_detected === true;

    const className =
        data.class_name ||
        "No object detected";

    const confidence =
        Number(
            data.confidence || 0
        );


    const gate =
        data.gate || {};


    const report = `CAMOUFLAGE BREAKER
AI ANALYSIS REPORT
========================================

STATUS
----------------------------------------
${detected
    ? "OBJECT DETECTED"
    : "NO OBJECT DETECTED"}

OBJECT
----------------------------------------
${className}

CONFIDENCE
----------------------------------------
${confidence.toFixed(2)}%

AI PIPELINE
----------------------------------------
SINet-V2 + ResNet50

SEGMENTATION DETAILS
----------------------------------------
Area Ratio       : ${
    gate.area_ratio !== undefined
        ? Number(gate.area_ratio).toFixed(4)
        : "N/A"
}

Largest Component: ${
    gate.largest_component_ratio !== undefined
        ? Number(gate.largest_component_ratio).toFixed(4)
        : "N/A"
}

Mean Probability : ${
    gate.mean_probability !== undefined
        ? Number(gate.mean_probability).toFixed(4)
        : "N/A"
}

Maximum Probability : ${
    gate.max_probability !== undefined
        ? Number(gate.max_probability).toFixed(4)
        : "N/A"
}

========================================

Generated by Camouflage Breaker
Deep Learning Computer Vision System
`;


    return new Blob(
        [report],
        {
            type: "text/plain"
        }
    );

}


/* ============================================================
   LOAD JSZIP
============================================================ */

async function loadJSZip() {

    if (
        window.JSZip
    ) {

        return window.JSZip;

    }


    return new Promise(
        (resolve, reject) => {

            const script =
                document.createElement(
                    "script"
                );


            script.src =
                "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";


            script.onload =
                () => {

                    if (window.JSZip) {

                        resolve(
                            window.JSZip
                        );

                    } else {

                        reject(
                            new Error(
                                "ZIP library failed to load."
                            )
                        );

                    }

                };


            script.onerror =
                () => {

                    reject(
                        new Error(
                            "Unable to load ZIP download system."
                        )
                    );

                };


            document.head.appendChild(
                script
            );

        }
    );

}


/* ============================================================
   DOWNLOAD COMPLETE ANALYSIS
============================================================ */

downloadButton.addEventListener(
    "click",
    async () => {

        if (!currentResult) {

            return;

        }


        const originalText =
            downloadButton.innerHTML;


        try {

            downloadButton.disabled =
                true;

            downloadButton.innerHTML =
                "<span>...</span> PREPARING ZIP";


            const JSZip =
                await loadJSZip();


            const zip =
                new JSZip();


            const folder =
                zip.folder(
                    "Camouflage_Breaker_Analysis"
                );


            const result =
                currentResult;


            const files = [

                {
                    name:
                        "original.jpg",

                    data:
                        result.original
                },

                {
                    name:
                        "ai_overlay.jpg",

                    data:
                        result.overlay
                },

                {
                    name:
                        "boundary.jpg",

                    data:
                        result.boundary
                },

                {
                    name:
                        "extracted_object.jpg",

                    data:
                        result.crop
                },

                {
                    name:
                        "mask.png",

                    data:
                        result.mask
                }

            ];


            files.forEach(
                (file) => {

                    if (!file.data) {

                        return;

                    }


                    const cleanBase64 =
                        file.data.includes(",")
                            ? file.data.split(",")[1]
                            : file.data;


                    folder.file(
                        file.name,
                        cleanBase64,
                        {
                            base64: true
                        }
                    );

                }
            );


            folder.file(
                "analysis.txt",
                createTextBlob(
                    result
                )
            );


            const zipBlob =
                await zip.generateAsync(
                    {
                        type: "blob",
                        compression:
                            "DEFLATE",
                        compressionOptions:
                            {
                                level: 6
                            }
                    }
                );


            const url =
                URL.createObjectURL(
                    zipBlob
                );


            const link =
                document.createElement(
                    "a"
                );


            link.href =
                url;

            link.download =
                "Camouflage_Breaker_Analysis.zip";


            document.body.appendChild(
                link
            );

            link.click();

            link.remove();


            setTimeout(
                () => {

                    URL.revokeObjectURL(
                        url
                    );

                },
                1000
            );


        } catch (error) {

            showError(
                "DOWNLOAD FAILED",
                error.message ||
                "Unable to create the analysis ZIP file."
            );


        } finally {

            downloadButton.disabled =
                false;

            downloadButton.innerHTML =
                originalText;

        }

    }
);


/* ============================================================
   NEW SCAN
============================================================ */

newScanButton.addEventListener(
    "click",
    () => {

        resetApplication();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);


/* ============================================================
   RESET
============================================================ */

function resetApplication() {

    selectedFile = null;

    currentResult = null;

    fileInput.value = "";

    previewImage.removeAttribute(
        "src"
    );

    originalResult.removeAttribute(
        "src"
    );

    overlayResult.removeAttribute(
        "src"
    );

    boundaryResult.removeAttribute(
        "src"
    );

    cropResult.removeAttribute(
        "src"
    );

    uploadContent.style.display =
        "flex";

    previewArea.style.display =
        "none";

    processingPanel.style.display =
        "none";

    resultsSection.style.display =
        "none";

    detectButton.disabled =
        true;

    fileStatus.textContent =
        "";

    progressFill.style.width =
        "0%";

    confidenceFill.style.width =
        "0%";

    confidenceValue.textContent =
        "0%";

    confidenceValueLarge.textContent =
        "0%";

    predictionName.textContent =
        "—";

    predictionTitle.textContent =
        "—";

    resultBadge.textContent =
        "OBJECT DETECTED";

    hideError();

}


/* ============================================================
   ERROR
============================================================ */

function showError(
    title,
    message
) {

    errorTitle.textContent =
        title;

    errorMessage.textContent =
        message;

    errorPanel.style.display =
        "flex";

    errorPanel.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


function hideError() {

    errorPanel.style.display =
        "none";

}


/* ============================================================
   INITIAL STATE
============================================================ */

processingPanel.style.display =
    "none";

resultsSection.style.display =
    "none";

errorPanel.style.display =
    "none";