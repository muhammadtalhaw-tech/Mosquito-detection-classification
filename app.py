
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow import keras

# =====================================
# MOSQUITO IMAGE CLASSIFICATION WEBSITE
# =====================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mosquito_detector.keras"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 70.0

CLASS_NAMES = [
    "NOT MOSQUITO",
    "DENGUE-VECTOR SPECIES",
    "NON-DENGUE MOSQUITO"
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# =====================================
# LOAD SAVED MODEL
# =====================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Check that models/mosquito_detector.keras exists."
    )

model = keras.models.load_model(MODEL_PATH)


# =====================================
# IMAGE HELPERS
# =====================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Model includes MobileNetV2 preprocessing.
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    probabilities = [
        {
            "name": CLASS_NAMES[i],
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in range(len(CLASS_NAMES))
    ]

    return predicted_index, confidence, probabilities


# =====================================
# MAIN WEBSITE
# =====================================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None
    probabilities = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please choose an image first."

        else:
            file = request.files["image"]

            if file.filename == "":
                error = "Please choose an image first."

            elif not allowed_file(file.filename):
                error = (
                    "Invalid file type. Upload JPG, JPEG, PNG, "
                    "BMP, or WEBP."
                )

            else:
                try:
                    safe_name = secure_filename(file.filename)

                    if not safe_name:
                        safe_name = "uploaded_image"

                    unique_name = f"{uuid4().hex}_{safe_name}"
                    save_path = UPLOAD_FOLDER / unique_name

                    file.save(save_path)

                    # Validate uploaded image.
                    with Image.open(save_path) as img:
                        img.verify()

                    predicted_index, confidence, probabilities = (
                        predict_image(save_path)
                    )

                    image_url = f"/static/uploads/{unique_name}"

                    # =====================================
                    # CONFIDENCE CHECK
                    # =====================================

                    if confidence < CONFIDENCE_THRESHOLD:
                        result = {
                            "title": "UNKNOWN / NOT SURE",
                            "message": (
                                "The model is not confident enough "
                                "to classify this image. Please upload "
                                "a clear mosquito image."
                            ),
                            "confidence": round(confidence, 2),
                            "kind": "unknown"
                        }

                    elif predicted_index == 0:
                        result = {
                            "title": "NOT MOSQUITO",
                            "message": (
                                "This image was classified as "
                                "not a mosquito."
                            ),
                            "confidence": round(confidence, 2),
                            "kind": "not-mosquito"
                        }

                    elif predicted_index == 1:
                        result = {
                            "title": "DENGUE-VECTOR SPECIES",
                            "message": (
                                "The image was classified as a "
                                "mosquito species associated with "
                                "dengue transmission."
                            ),
                            "confidence": round(confidence, 2),
                            "kind": "mosquito"
                        }

                    else:
                        result = {
                            "title": "NON-DENGUE MOSQUITO",
                            "message": (
                                "The image was classified as another "
                                "mosquito species in the training dataset."
                            ),
                            "confidence": round(confidence, 2),
                            "kind": "mosquito"
                        }

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError
                ):
                    error = "The uploaded file is not a valid image."

                except Exception as e:
                    error = f"Prediction error: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=image_url,
        probabilities=probabilities
    )


# =====================================
# FILE SIZE ERROR
# =====================================

@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Maximum file size is 8 MB.",
        image_url=None,
        probabilities=None
    ), 413


# =====================================
# RUN WEBSITE
# =====================================

if __name__ == "__main__":
    app.run(debug=True)