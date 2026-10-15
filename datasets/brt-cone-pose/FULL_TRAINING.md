# Full cone pose training flow

Started October 10, 2026. Training and evaluation completed October 11, 2026 at 07:17 EDT after 20 total epochs.
Benchmark checkpoints are compatibility checks, not usable cone pose models.

## Configuration

- Fresh YOLO26m pose training initialized from pretrained `yolo26m-pose.pt`.
- NVIDIA RTX 5060, CUDA device 0, mixed precision, 960-pixel input.
- Batch 4, verified with a 64-image benchmark including the most crowded scenes.
  Clean benchmark peak reserved CUDA memory: 5.59 GiB. Batch 8 ran out of memory.
- Up to 20 total epochs (user limit updated October 11); stop after 30 epochs without improved validation pose mAP.
- AdamW, learning rate 0.0015, cosine decay to 2% of initial rate, three warmup epochs.
- Checkpoint selection uses pose mAP50–95 rather than a sum of pose and box metrics.
- Small online mosaic probability (0.15), modest scale/translation and color jitter;
  no flips, mixup, or cutmix. Mosaic closes for the last 5 configured epochs.

## Offline augmentation

`augment_brt_pose.py` builds a training manifest of 30,000 images: all 9,218
original training images and 20,782 deterministic variants. Each source gets
two or three variants. Training sources are uniformly balanced across variants.
Variants retain up to 1600 pixels on the long edge to preserve small cones.

Transformations include rotations up to 10 degrees, 0.92–1.04 scale, small
translations and perspective perturbations, mild radial lens distortion,
directional motion blur, contrast/exposure changes, noise, and JPEG compression.
Boxes use transformed sampled perimeters; labeled keypoints receive the same
homography and radial mapping as pixels. Missing points remain missing. Offscreen
points are masked, and heavily clipped cones are removed. If geometry removes
every cone from a non-background source, that variant falls back to the original
geometry. Background images remain background images.

Validation and test images are not augmented. `split-overlap.json` found zero
exact image duplicates within or across the original splits; temporal/near-duplicate
overlap was not ruled out. Augmentation parameters and sources are stored in
`augmented-30k/provenance.json`. The supervisor verifies all 30,000 manifest entries
and numeric label constraints before training and writes `augmentation-audit.json`.

## Progress and continuation

`run_brt_training_flow.py` runs under the systemd user service
`brt-cone-pose-training.service` and waits for augmentation completion before
starting full training and evaluation. A sleep inhibitor keeps the machine awake
during GPU training and evaluation. Progress is recorded in:

- `augmentation.log`
- `training-flow.log`
- `full-training.log`
- `full-training-status.json`
- `../../runs/brt-pose-m-960-30k/results.csv`

The T3 monitoring task was disabled after final training, evaluation, inference
verification and report delivery on October 11, 2026.

If interrupted after checkpoints exist, resume using:

```sh
uv run --locked train_brt_full.py --resume runs/brt-pose-m-960-30k/weights/last.pt --batch 4 --epochs 20 --workers 6
```

## Final evaluation and outputs

The flow will save the best checkpoint to
`../../runs/brt-pose-m-960-30k/weights/best.pt`, plus test box/pose metrics,
strict validation and test keypoint error reports, and annotated held-out examples.
PCK at 5% and 10% of ground-truth box height requires a same-class box match with
IoU at least 0.5 and keypoint confidence at least 0.5. End-to-end PCK includes
missed cones as failures; matched-only PCK is reported separately. Pixel error
and per-keypoint results help assess actual localization rather than relying only
on the comparatively permissive default pose metric for custom keypoint layouts.

After the final model is verified, infer boxes and keypoints with:

```sh
uv run --locked cone_pose_predict.py path/to/image-or-video --output pose-predictions
```

This writes annotated images/frames and `predictions.jsonl`, containing original
pixel boxes and eight `(x, y, confidence)` triples per cone. Keypoints 6–7 are
supervised only for large orange cones. No numeric accuracy is claimed until
full training and evaluation finish.

## Speed tests — October 11, 2026

Training was paused after epoch 2, with a backup at
`runs/brt-pose-m-960-30k/weights/pre-speed-optimization.pt`. Any partial epoch 3
work is replayed from that checkpoint. The resumed run keeps optimizer/EMA state.
The cosine schedule now targets 20 total epochs; mosaic closes at epoch 16.
An explicit resume override ensures the checkpoint's old 180-epoch limit is not restored.

Isolated 256-image tests at 960 pixels included the most crowded source images.
Steady throughput excludes the first 16 batches and synchronizes CUDA:

- Batch 4, six workers, original automatic channels-last: 22.95 images/s, 5.67 GiB peak reserved.
- Batch 4, ten workers: 22.78 images/s, 5.67 GiB.
- Batch 5, six workers: 21.52 images/s, 6.66 GiB.
- Compilation, batch 4, six workers: 18.92 images/s, 4.32 GiB; 157 seconds total including compilation.
- Exploratory NCHW layout: 18.06 images/s; not the original training layout.

Retained batch 4, six workers, automatic channels-last, mixed precision and no compilation.
No tested alternative improved steady throughput. These short original-image benchmarks
are comparative tests, not guaranteed full augmented-dataset epoch times. No quality gain
or quality equivalence is inferred from their one-epoch accuracy. Training model, image
resolution, dataset and geometric augmentation are retained. Accuracy still requires
full held-out evaluation; the 20-epoch cap may limit final convergence.
Raw results: `speed-benchmarks.json` and `speed-*.log`.

## Final measured results

Deliverable: `runs/brt-pose-m-960-30k/weights/best.pt` (relative to repository root),
selected by validation pose mAP; epoch 20 was best at 0.92701. Test data did not
participate in checkpoint selection or hyperparameter tuning. Original detector
checkpoints were preserved. No training beyond the requested 20 epochs was run.

Untouched test box mAP50–95: **0.86884**; pose mAP50–95: **0.91980**.
Strict results at detection confidence 0.25, point confidence 0.5, same-class
one-to-one box IoU at least 0.5:

- Validation end-to-end PCK: **77.50%** within 5% of box height; **90.00%** within 10%.
- Test end-to-end PCK: **77.60%** within 5%; **90.21%** within 10%.
- Test matched-cone PCK: **83.11%** within 5%; **96.61%** within 10%.
- Test median matched labeled-point error: **1.14 original-image pixels**, or **2.38%** of box height.
- Test cone recall: **93.35%** (9679 of 10369); precision: **87.02%** at confidence 0.25.

End-to-end PCK includes missed cones and low-confidence keypoints as failures.
Median error only includes matched cones and must not be read as an all-cone guarantee.
The default custom-keypoint OKS metric is lenient; strict PCK better represents
localization. Validation and test strict results agree closely. The validation
curve largely flattened in the final epochs (0.92665 at epoch 15 to 0.92701 at 20).
This is a useful initial pose model, but ~22.4% of labeled test points fail the
5%-height criterion. Small/distant, occluded, and clipped cones remain practical
limitations; an inspected validation overlay misses a foreground clipped cone.
No per-size/occlusion breakdown was measured, so those cases are not quantified.
Points 6–7 are supervised only on large orange cones; ignore those outputs on other classes.
Exact split duplicates were ruled out, but near-duplicate/temporal split leakage
and generalization to this car's camera remain unverified. Further work should
use new car-camera validation data and inspect failures before choosing refinement.
Additional training requires a new user instruction under the current epoch cap.

Reports: `runs/brt-pose-m-960-30k/{validation-pose-accuracy.json,pose-accuracy.json,test-metrics.json}`.
Annotated examples: `runs/brt-pose-m-960-30k/{val-examples,test-examples}`.
Inference was smoke-tested on a validation image using `cone_pose_predict.py`.

```sh
uv run --locked cone_pose_predict.py path/to/image-or-video --output pose-predictions
```

The default checkpoint is the final best pose model. Output is per-image/frame
JPEG overlays and `predictions.jsonl`, with original-pixel `box_xyxy`, detection
class/confidence, and eight `keypoints_xy_confidence` triples per cone. Video
input exports frames rather than an assembled annotated video.
