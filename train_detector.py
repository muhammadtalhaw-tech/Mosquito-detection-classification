
# ============================================================
# MOSQUITO CLASSIFIER - TRAINING
# Classes:
# 0 = NOT MOSQUITO
# 1 = DENGUE-VECTOR SPECIES
# 2 = NON-DENGUE MOSQUITO
# ============================================================

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# SETTINGS
# =========================

SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
MAX_PER_CLASS = 3000

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

MODEL_PATH = BASE_DIR / "mosquito_model.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# IMAGE LOADER
# =========================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    folder = Path(folder)

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        return []

    files = [
        str(path)
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    return files


# =========================
# LOAD DATASET FILES
# =========================

print("\nLoading dataset files...")

dengue_files = get_images(DATASET_DIR / "dengue")
other_files = get_images(DATASET_DIR / "non_dengue")

cifar10_files = get_images(
    DATASET_DIR / "not_mosquito_50000"
)

cifar100_files = get_images(
    DATASET_DIR / "not_mosquito_extra" / "cifar100_images"
)

print("Dengue-vector source images:", len(dengue_files))
print("Other mosquito source images:", len(other_files))
print("CIFAR-10 non-mosquito images:", len(cifar10_files))
print("CIFAR-100 non-mosquito images:", len(cifar100_files))


# =========================
# BALANCE DATASET
# =========================

print("\nBalancing dataset...")

random.shuffle(cifar10_files)
random.shuffle(cifar100_files)

per_source = MAX_PER_CLASS // 2

not_files = (
    cifar10_files[:per_source]
    + cifar100_files[:per_source]
)

# Fill any shortage from remaining images
if len(not_files) < MAX_PER_CLASS:
    remaining = MAX_PER_CLASS - len(not_files)

    leftovers = (
        cifar10_files[per_source:]
        + cifar100_files[per_source:]
    )

    random.shuffle(leftovers)
    not_files.extend(leftovers[:remaining])

random.shuffle(dengue_files)
random.shuffle(other_files)

dengue_files = dengue_files[:MAX_PER_CLASS]
other_files = other_files[:MAX_PER_CLASS]

# Keep all classes equal
n = min(
    len(not_files),
    len(dengue_files),
    len(other_files)
)

not_files = not_files[:n]
dengue_files = dengue_files[:n]
other_files = other_files[:n]

print("\n========== FINAL DATASET ==========")
print("NOT MOSQUITO:", len(not_files))
print("DENGUE-VECTOR SPECIES:", len(dengue_files))
print("NON-MOSQUITO SPECIES:", len(other_files))
print("USING PER CLASS:", n)

if n == 0:
    raise ValueError(
        "No images found. Check dataset folder names and image files."
    )

if n < MAX_PER_CLASS:
    print(
        "\nWARNING: One or more classes have fewer than "
        f"{MAX_PER_CLASS} usable source images."
    )


# =========================
# LABELS
# =========================

all_paths = (
    not_files
    + dengue_files
    + other_files
)

all_labels = (
    [0] * n
    + [1] * n
    + [2] * n
)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels, dtype=np.int32)


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=all_labels
)

print("\nTRAINING IMAGES:", len(train_paths))
print("VALIDATION IMAGES:", len(val_paths))


# =========================
# IMAGE PREPROCESSING
# =========================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    return image, label


# =========================
# TF DATASETS
# =========================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)


# =========================
# BUILD MODEL
# =========================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(
    3,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# =========================
# COMPILE
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# CALLBACKS
# =========================

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)


# =========================
# TRAIN
# =========================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# =========================
# EVALUATE CURRENT MODEL
# =========================

print("\nEvaluating final in-memory model...")

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")


# =========================
# CLASSIFICATION REPORT
# =========================

print("\nGenerating predictions...")

y_true = val_labels

predictions = model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

class_names = [
    "NOT MOSQUITO",
    "DENGUE-VECTOR SPECIES",
    "NON-DENGUE MOSQUITO"
]

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )
)


# =========================
# SAVE FINAL MODEL
# =========================

# The checkpoint already saves the best val_accuracy model.
# Save the final in-memory model separately for reference.
final_model_path = BASE_DIR / "mosquito_model_final.keras"

model.save(final_model_path)

print("\nTraining complete.")
print("Best checkpoint:", MODEL_PATH)
print("Final in-memory model:", final_model_path)