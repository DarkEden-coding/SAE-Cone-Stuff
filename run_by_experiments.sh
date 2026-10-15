#!/usr/bin/env bash
# A: nano 10 epochs on cleaned blue/yellow. B: +10 on label-adjusted. C (control): +10 on cleaned.
set -euo pipefail
cd "$(dirname "$0")"
D=datasets/brt-cone-pose
run() { [ -f "runs/$1/weights/last.pt" ] && [ -f "runs/$1/.done" ] && return 0
        uv run --locked train_by_experiment.py --name "$1" "${@:2}" > "runs/$1.log" 2>&1; touch "runs/$1/.done"; }
run by-n-A-clean --data $D/brt-cone-pose-by-clean/data.yaml
run by-n-B-adj   --data $D/brt-cone-pose-by-clean-adj/data.yaml --model runs/by-n-A-clean/weights/last.pt --finetune
run by-n-C-clean --data $D/brt-cone-pose-by-clean/data.yaml     --model runs/by-n-A-clean/weights/last.pt --finetune
echo all done
