#!/bin/sh
# Export the cone pose model to CoreML for Live mode (landscape 1280x736 and portrait 736x1280, FP16,
# raw one-to-many head: NMS runs in the app, matching the server's predict path).
# Uses apps/local-detector's environment, which pins a coremltools-compatible torch.
# Usage (repo root): sh ios/ConeCapture/scripts/export_model.sh [weights.pt]
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WEIGHTS=${1:-$ROOT/runs/by-n-E-far/weights/best.pt}
OUT=$ROOT/ios/ConeCapture/ConeCapture/Resources
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$WEIGHTS" "$WORK/conepose.pt"
cd "$ROOT/apps/local-detector"
uv run --locked python - "$WORK/conepose.pt" "$OUT" <<'PY'
import shutil, sys
from pathlib import Path
from ultralytics import YOLO
weights, out = Path(sys.argv[1]), Path(sys.argv[2])
for h, w in ((736, 1280), (1280, 736)):
    path = Path(YOLO(str(weights)).export(format="coreml", imgsz=[h, w], half=True))
    target = out / f"ConePose{w}x{h}.mlpackage"
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(path), target)
    print("wrote", target)
PY
cp "$ROOT/cone_app/assets/kp_correction_E_1920.json" "$OUT/kp_correction.json"
