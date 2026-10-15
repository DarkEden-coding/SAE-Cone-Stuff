"""Train a cone pose model; --smoke checks a tiny subset before a full run."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/brt-cone-pose/data.local.yaml"
DETECTOR = ROOT / "runs/yolo26m-640-small-cones/weights/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-detector", action="store_true",
                        help="Initialize YOLO26m pose with the existing cone detector's matching weights")
    parser.add_argument("--smoke", action="store_true", help="One epoch on 16 train / 4 validation images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = yaml.safe_load(DATA.read_text())
    data = DATA
    if args.smoke:
        directory = DATA.parent / "smoke"
        directory.mkdir(exist_ok=True)
        dataset = Path(config["path"])
        for split, count in (("train", 16), ("val", 4)):
            images = sorted((dataset / "images" / split).glob("*.jpg"))[:count]
            if len(images) != count:
                raise ValueError(f"Insufficient {split} images")
            manifest = directory / f"{split}.txt"
            manifest.write_text("\n".join(str(p.resolve()) for p in images) + "\n")
            config[split] = str(manifest.resolve())
        data = directory / "data.yaml"
        data.write_text(yaml.safe_dump(config, sort_keys=False))
    # Keeping the pose architecture is essential: loading a detect .pt directly
    # would continue box-only training rather than add a keypoint head.
    model = (
        YOLO("yolo26m-pose.yaml").load(str(DETECTOR))
        if args.from_detector else YOLO("yolo26n-pose.pt")
    )
    name = "brt-pose-m-transfer" if args.from_detector else "brt-pose-n"
    if args.smoke:
        name += "-smoke"
    model.train(
        data=str(data), epochs=1 if args.smoke else args.epochs,
        batch=2 if args.smoke else args.batch, imgsz=args.imgsz,
        device=args.device, workers=2, project=str(ROOT / "runs"), name=name,
        # The published YAML has no left/right keypoint permutation. Do not
        # invent one; disable flips until the annotation semantics are verified.
        fliplr=0.0, flipud=0.0, plots=False,
        mosaic=0.0 if args.smoke else 1.0,
    )
    if args.smoke:
        best = YOLO(str(Path(model.trainer.save_dir) / "weights/best.pt"))
        image = Path(config["val"]).read_text().splitlines()[0]
        result = best.predict(image, conf=0.001, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
        assert best.task == "pose"
        assert list(best.model.model[-1].kpt_shape) == [8, 3]
        assert result.keypoints is not None
        assert result.keypoints.data.shape[1:] == (8, 3)
        assert len(result.boxes) == len(result.keypoints)
        print(f"Smoke passed: boxes {tuple(result.boxes.xyxy.shape)}, "
              f"keypoints {tuple(result.keypoints.data.shape)}")


if __name__ == "__main__":
    main()
