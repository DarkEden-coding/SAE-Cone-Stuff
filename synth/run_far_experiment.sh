#!/usr/bin/env bash
# Far-cone experiment (run E): wait for synth_train rendering/postprocess, build native-res tiles, fine-tune run D,
# evaluate long-range PnP at 1920 on the synth_data test set. Log: synth_train/experiment.log
set -euo pipefail
cd "$(dirname "$0")/.."
until grep -q DONE synth_train/postprocess.log 2>/dev/null; do sleep 60; done
uv run --locked synth/build_far_dataset.py synth_train
uv run --locked train_by_experiment.py --data datasets/synth-far/data.yaml --name by-n-E-far \
  --model runs/by-n-D-aug/weights/last.pt --finetune --epochs 20
uv run --locked synth/far_eval.py synth_data D_aug30=runs/by-n-D-aug/weights/last.pt \
  E_far=runs/by-n-E-far/weights/best.pt --real-val --out synth_data/eval/far_eval.json
echo EXPERIMENT_DONE
