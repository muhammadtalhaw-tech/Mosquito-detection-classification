
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# MOSQUITO DETECTOR - MODEL TESTING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mosquito_detector.keras"

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "NOT MOSQUITO",
    "DENGUE-ASSOCIATED MOSQUITO",
    "NON-DENGUE MOSQUITO"
]

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading mosquito detection model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Pehle training complete karo."
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model path:", MODEL_PATH)

# ==========================================
# GET IMAGE PATH
# ==========================================

image_path = input(
    "\nEnter full image path: "
).strip().strip('"')

image_file = Path(image_path)

if not image_file.is_file():
    print("\nERROR: Image file not found!")
    raise SystemExit

# ==========================================
# LOAD AND PREPROCESS IMAGE
# ==========================================

try:
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32)

    # SAME preprocessing as training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(img_array, axis=0)

except Exception as e:
    print("\nERROR loading image:", e)
    raise SystemExit

# ==========================================
# PREDICT
# ==========================================

print("\nAnalyzing image...")

predictions = model.predict(
    img_array,
    verbose=0
)[0]

predicted_index = int(np.argmax(predictions))
confidence = float(predictions[predicted_index]) * 100

# ==========================================
# DISPLAY ALL PROBABILITIES
# ==========================================

print("\n" + "=" * 45)
print("       MOSQUITO DETECTION RESULT")
print("=" * 45)

for i, name in enumerate(CLASS_NAMES):
    probability = float(predictions[i]) * 100
    print(f"{name}: {probability:.2f}%")

print("-" * 45)
print("FINAL PREDICTION:", CLASS_NAMES[predicted_index])
print(f"CONFIDENCE: {confidence:.2f}%")
print("=" * 45)

# ==========================================
# SIMPLE RESULT MESSAGE
# ==========================================

if predicted_index == 0:
    print("\nRESULT: This image is classified as NOT MOSQUITO.")

elif predicted_index == 1:
    print(
        "\nRESULT: This image is classified as "
        "DENGUE-ASSOCIATED MOSQUITO."
    )

elif predicted_index == 2:
    print(
        "\nRESULT: This image is classified as "
        "NON-DENGUE MOSQUITO."
    )

print(
    "\nNOTE: These are model predictions based on folder labels. "
    "They do not confirm whether a mosquito is infected with dengue."
)