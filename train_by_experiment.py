"""Short nano pose runs for the cleaned / label-adjusted blue-yellow experiments.

uv run --locked train_by_experiment.py --data DATA_YAML --name RUN [--model yolo26n-pose.pt] [--epochs 10] [--finetune]

Hyperparameters match the original nano BRT run (batch 4, 960 px, AdamW, cosine LR, light mosaic/scale/HSV,
seed 26), and checkpoints are selected on validation pose mAP50-95 as in train_brt_full.py. --finetune
continues from an earlier run's weights with a lower peak LR and a short warmup.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from train_brt_full import KeypointTrainer

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', default=str(ROOT / 'yolo26n-pose.pt'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--finetune', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is required')
    torch.set_num_threads(6)
    YOLO(args.model).train(
        trainer=KeypointTrainer, data=args.data, device=0, epochs=args.epochs, patience=30,
        batch=4, imgsz=960, workers=args.workers, cache=False,
        project=str(ROOT / 'runs'), name=args.name, exist_ok=False,
        optimizer='AdamW', lr0=0.0005 if args.finetune else 0.0015, lrf=0.02, cos_lr=True,
        warmup_epochs=0.5 if args.finetune else 3, weight_decay=0.0005,
        fliplr=0, flipud=0, degrees=0, perspective=0, shear=0, translate=0.03, scale=0.15,
        mosaic=0.15, close_mosaic=5, mixup=0, cutmix=0, hsv_h=0.005, hsv_s=0.25, hsv_v=0.25,
        seed=26, deterministic=True, amp=True, plots=False,
    )


if __name__ == '__main__':
    main()
