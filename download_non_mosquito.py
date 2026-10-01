
from pathlib import Path
from tensorflow.keras.datasets import cifar10
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset" / "not_mosquito_50000"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading complete CIFAR-10 training dataset...")

(x_train, y_train), (_, _) = cifar10.load_data()

print("Saving 50,000 images...")

for i, image_array in enumerate(x_train, start=1):
    path = OUTPUT_DIR / f"non_mosquito_{i:05d}.png"
    if not path.exists():
        Image.fromarray(image_array).save(path)

    if i % 1000 == 0:
        print(f"Saved/checked: {i}/50000")

print("DONE!")
print("Total dataset images:", len(x_train))
print("Folder:", OUTPUT_DIR)