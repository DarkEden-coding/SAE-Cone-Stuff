# BRT cone pose training assessment

Downloaded and checked on October 10, 2026. The original archive is retained as
`dataset.zip`; extraction is under `brt-cone-pose-11k/`. Original BRT instructions,
license, and repository commit metadata are preserved in `source/`.

## Data verification

The archive extracted successfully with ZIP CRC verification. `audit.json` records
the archive SHA-256, label checks, and per-split counts. There are 9,218 training,
1,152 validation, and 1,153 test images, with matching label files and 105,662
annotated cones. All object rows passed checks for 29 fields, finite values, valid
classes, normalized coordinates, positive box sizes, and keypoint visibility.
This is a numeric label audit, not a review of annotation accuracy or a complete
image decoding check.

Labels already contain both boxes and eight keypoints in Ultralytics pose format.
`data.local.yaml` fixes the original YAML's relative dataset root for this workspace.
Class IDs are BLUE=0, ORANGE_BIG=1, ORANGE=2, UNDEFINED=3, YELLOW=4. There are
no UNDEFINED annotations. Keypoints 0–5 are visible on all cones; 6–7 are visible
only on the 5,189 large orange cones. Other cones have those two points marked
missing. The README does not define anatomical names for these eight indices;
inspect annotated examples before using them for geometric localization.

## Feasibility and verification

Training is straightforward with the installed Ultralytics 8.4.163. A pose model
outputs boxes, classes, and keypoints together. Existing `detect` checkpoints
cannot acquire keypoint outputs by continuing ordinary detector training.
Instead, construct a YOLO26m pose architecture and load matching weights from
the existing multiclass detector. Direct transfer into a five-class, eight-point
architecture matched 756/939 state tensors, covering about 92.5% of state elements.
This does not mean that existing class outputs or keypoint predictions transfer:
class names/order differ, and the keypoint branch is new.

The following command passed on this machine's RTX 5060 (8 GB):

```sh
uv run --locked train_brt_pose.py --from-detector --smoke
```

It trained for one epoch on 16 images, validated on four, saved and reloaded a
checkpoint, and verified box outputs shaped `(N, 4)` and keypoints `(N, 8, 3)`.
Reported training GPU memory was about 1.47 GB at batch 2 / 640 pixels.
See `smoke-training.log` and `../../runs/brt-pose-m-transfer-smoke/`.
This establishes pipeline compatibility, not useful accuracy: smoke pose mAP
was zero. The smoke checkpoint must not be deployed as a trained replacement.

## Full training options

Reuse the existing cone detector's features:

```sh
uv run --locked train_brt_pose.py --from-detector --epochs 100 --batch 4
```

Start a smaller model using pretrained YOLO26n pose weights (downloads weights
on first use):

```sh
uv run --locked train_brt_pose.py --epochs 100 --batch 4
```

Both use 640 pixels by default; evaluate 800 or 960 for distant cones, with a
smaller batch if needed. Horizontal/vertical flipping is disabled because BRT
does not supply the keypoint permutation required for correct flipping.
Full training has not been started. Evaluate the eventual checkpoint on held-out
BRT test images and the existing project's test images to check pose accuracy
and retention of blue/yellow detection quality. The old dataset has box labels
only, so it cannot be merged directly as fully supervised pose training data.

The current web viewer already calls `result.plot()`, which can draw pose points,
and accepts alternate multiclass checkpoints through `--model`. Exporting numeric
keypoints through the UI would require adding an output; Python inference exposes
them through `result.keypoints.xy` and `result.keypoints.conf`. No live app or
existing detector checkpoint was replaced.

Credit Bauman Racing Team for annotations and FSOCO for source images. Annotation
licensing and original image terms are described separately in `source/README.md`.
