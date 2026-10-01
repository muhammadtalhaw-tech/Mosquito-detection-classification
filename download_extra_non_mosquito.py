
from pathlib import Path
from tensorflow.keras.datasets import cifar100
from PIL import Image
import numpy as np

# Project folders
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset" / "not_mosquito_extra" / "cifar100_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-100 dataset...")
(x_train, _), (x_test, _) = cifar100.load_data(
    label_mode="fine"
)

images = np.concatenate([x_train, x_test], axis=0)

print("Total images:", len(images))
print("Saving images...")

saved = 0

for i, image_array in enumerate(images):
    output_path = OUTPUT_DIR / f"non_mosquito_{i:05d}.jpg"

    if output_path.exists():
        continue

    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(output_path, format="JPEG", quality=90)
    saved += 1

    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1}/{len(images)}")

print("\nDOWNLOAD COMPLETE")
print("New images saved:", saved)
print("Folder:", OUTPUT_DIR)
print("Total files now:", len(list(OUTPUT_DIR.glob("*.jpg"))))