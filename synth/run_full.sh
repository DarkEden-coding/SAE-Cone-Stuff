#!/usr/bin/env bash
# Render the full synthetic set and post-process frames as they finish. Safe to re-run (resumes).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-synth_data}
COUNT=${COUNT:-1000}
B=/mnt/Primary/tools/blender-4.5.14-linux-x64/blender
mkdir -p "$OUT"
"$B" -b -P synth/render_blender.py -- --out "$OUT" --start 0 --count "$COUNT" --samples 32 --skip-existing \
  2>&1 | grep --line-buffered -E '\[synth\]|Error|Traceback' > "$OUT/render.log" &
RENDER=$!
while kill -0 "$RENDER" 2>/dev/null; do
  uv run synth/postprocess.py "$OUT" --viz >> "$OUT/postprocess.log" 2>&1 || true
  sleep 20
done
wait "$RENDER"
uv run synth/postprocess.py "$OUT" --viz >> "$OUT/postprocess.log" 2>&1
echo "done: $(ls "$OUT/images" | wc -l) images" | tee -a "$OUT/postprocess.log"
