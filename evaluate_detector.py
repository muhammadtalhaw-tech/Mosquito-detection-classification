
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================
# EVALUATE SAVED MOSQUITO MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "mosquito_detector.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
MAX_PER_CLASS = 3000

CLASS_NAMES = [
    "NOT MOSQUITO",
    "DENGUE-ASSOCIATED MOSQUITO",
    "NON-DENGUE MOSQUITO"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_images(folder):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

not_files = get_images(DATASET_DIR / "not_mosquito_50000")
dengue_files = get_images(DATASET_DIR / "dengue")
other_files = get_images(DATASET_DIR / "non_dengue")

print("Dataset counts:", len(not_files), len(dengue_files), len(other_files))

if min(len(not_files), len(dengue_files), len(other_files)) == 0:
    raise ValueError("A class folder is empty. Check dataset paths.")

n = min(len(not_files), len(dengue_files), len(other_files), MAX_PER_CLASS)

random.shuffle(not_files)
random.shuffle(dengue_files)
random.shuffle(other_files)

not_files = not_files[:n]
dengue_files = dengue_files[:n]
other_files = other_files[:n]

all_paths = not_files + dengue_files + other_files
all_labels = [0] * n + [1] * n + [2] * n

_, paths_val, _, y_val = train_test_split(
    all_paths,
    all_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=all_labels
)

print("Validation images:", len(paths_val))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

ds = tf.data.Dataset.from_tensor_slices(
    ([str(p) for p in paths_val], y_val)
)
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)

print("\nEvaluating saved model...")
loss, accuracy = model.evaluate(ds, verbose=1)
print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Predictions
probabilities = model.predict(ds, verbose=1)
predictions = np.argmax(probabilities, axis=1)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print("Rows = actual class; columns = predicted class")
print(confusion_matrix(y_val, predictions, labels=[0, 1, 2]))

print("\nClass order:")
for i, name in enumerate(CLASS_NAMES):
    print(i, "=", name)