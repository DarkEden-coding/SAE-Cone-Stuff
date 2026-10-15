"""Create training-only zoomed-out copies of labeled cone images."""

from pathlib import Path
import random

import cv2


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset/train"
OUTPUT = ROOT / "dataset/small-cones"
SEED = 26
FRACTION = 0.25


def main() -> None:
    """Write a reproducible 25% supplement and an image list for YOLO training."""
    images = sorted((SOURCE / "images").glob("*.jpg"))
    rng = random.Random(SEED)
    selected = set(rng.sample(images, round(len(images) * FRACTION)))
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels").mkdir(parents=True, exist_ok=True)
    paths = [str(image.resolve()) for image in images]

    for image in images:
        if image not in selected:
            continue
        label = SOURCE / "labels" / f"{image.stem}.txt"
        rows = [line.split() for line in label.read_text().splitlines()]
        # A few source annotations are polygons; skip rather than changing their format.
        if any(len(row) != 5 for row in rows):
            continue
        original = cv2.imread(str(image))
        if original is None:
            raise ValueError(f"Cannot read {image}")
        height, width = original.shape[:2]
        scale = rng.uniform(0.55, 0.8)
        resized_width, resized_height = round(width * scale), round(height * scale)
        left, top = (width - resized_width) // 2, (height - resized_height) // 2
        canvas = cv2.resize(original, (resized_width, resized_height))
        padded = cv2.copyMakeBorder(
            canvas, top, height - resized_height - top, left, width - resized_width - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        target = OUTPUT / "images" / image.name
        if not cv2.imwrite(str(target), padded):
            raise OSError(f"Cannot write {target}")
        transformed = []
        for class_id, x, y, box_width, box_height in rows:
            x, y = float(x), float(y)
            new_x = (left + x * resized_width) / width
            new_y = (top + y * resized_height) / height
            new_width = float(box_width) * resized_width / width
            new_height = float(box_height) * resized_height / height
            assert 0 <= new_x - new_width / 2 <= new_x + new_width / 2 <= 1
            assert 0 <= new_y - new_height / 2 <= new_y + new_height / 2 <= 1
            transformed.append(f"{class_id} {new_x:.8f} {new_y:.8f} {new_width:.8f} {new_height:.8f}")
        (OUTPUT / "labels" / label.name).write_text("\n".join(transformed) + "\n")
        paths.append(str(target.resolve()))

    (OUTPUT / "train.txt").write_text("\n".join(paths) + "\n")
    (OUTPUT / "data.yaml").write_text(
        f"train: {OUTPUT / 'train.txt'}\n"
        f"val: {ROOT / 'dataset/valid/images'}\n"
        f"test: {ROOT / 'dataset/test/images'}\n"
        "names: ['blue_cone', 'yellow_cone']\n"
    )
    print(f"Training images: {len(images)} original + {len(paths) - len(images)} zoomed out")


if __name__ == "__main__":
    main()
