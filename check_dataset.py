
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "dataset"

folders = {
    "dengue": 5000,
    "non_dengue": 5000,
    "not_mosquito": 500
}

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\nMOSQUITO DENGUE AI - DATASET CHECK")
print("-" * 45)

for folder, target in folders.items():
    path = BASE_DIR / folder
    path.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

    print(f"\n{folder.upper()}")
    print(f"Images found: {len(images)}")
    print(f"Target: {target}")
    print(f"Remaining: {max(0, target - len(images))}")

print("\nDataset folder check complete.")