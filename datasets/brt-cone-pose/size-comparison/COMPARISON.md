# YOLO26 cone keypoints: model size comparison

Medium trained for 20 epochs; nano and small have a 40-epoch cap. Same 30,000-image train manifest, untouched validation/test splits, 960-pixel input, batch 4, seed 26, AdamW and augmentation settings. Medium was resumed after epoch 2 with its schedule changed from 180 to 20; the new runs use a 40-epoch schedule from the start. The epoch-20 comparison controls epoch count, but not learning-rate schedule or mosaic closing.

Checkpoint selection uses validation pose mAP50–95. Strict end-to-end PCK includes missed cones and low-confidence points. Categories are descriptive: within 1 percentage point of medium is comparable, 1–3 points lower is moderately lower, and more than 3 points lower is substantially lower. Above 1 point is higher. These are single-seed results, not statistical significance claims.

Small was stopped by user request after 31 completed epochs. The unfinished next epoch was discarded. Evaluation uses the saved best validation checkpoint; its 40-epoch learning-rate schedule was not shortened retroactively.

## Medium

Completed epochs: 20; best epoch: 20.
Best validation pose mAP50–95: 92.701%; box mAP50–95: 86.107%.
Best validation pose mAP through epoch 20: 92.701%.
Validation end-to-end PCK at 5% / 10% box height: 77.50% / 90.00%.
Test pose / box mAP50–95: 91.980% / 86.884%.
Test end-to-end PCK at 5% / 10%: 77.60% / 90.21%.
Parameters: 23,547,262; checkpoint: 45.48 MiB; batch-1 prediction median / p95: 17.87 / 22.92 ms.

## Nano

Completed epochs: 40; best epoch: 37.
Best validation pose mAP50–95: 91.810%; box mAP50–95: 84.397%.
Best validation pose mAP through epoch 20: 91.566%.
Validation end-to-end PCK at 5% / 10% box height: 75.78% / 90.42%.
Epoch-budget-20 pose mAP category: moderately lower accuracy (1–3 percentage points) (-1.135 percentage points).
Best-through-20 validation PCK at 5% / 10%: 74.90% / 90.03%.
Pose mAP category versus medium: comparable accuracy (within 1 percentage point) (-0.891 percentage points).
Strict keypoint category versus medium: moderately lower accuracy (1–3 percentage points) (-1.72 percentage points).
Test pose / box mAP50–95: 90.985% / 84.825%.
Test end-to-end PCK at 5% / 10%: 75.71% / 90.52%.
Parameters: 2,979,294; checkpoint: 6.16 MiB; batch-1 prediction median / p95: 10.06 / 12.24 ms.

## Small

Completed epochs: 31; best epoch: 31.
Best validation pose mAP50–95: 92.524%; box mAP50–95: 86.127%.
Best validation pose mAP through epoch 20: 92.387%.
Validation end-to-end PCK at 5% / 10% box height: 77.85% / 90.76%.
Epoch-budget-20 pose mAP category: comparable accuracy (within 1 percentage point) (-0.314 percentage points).
Best-through-20 validation PCK at 5% / 10%: 77.23% / 90.65%.
Pose mAP category versus medium: comparable accuracy (within 1 percentage point) (-0.177 percentage points).
Strict keypoint category versus medium: comparable accuracy (within 1 percentage point) (+0.35 percentage points).
Test pose / box mAP50–95: 92.012% / 86.799%.
Test end-to-end PCK at 5% / 10%: 77.84% / 90.80%.
Parameters: 10,747,838; checkpoint: 21.01 MiB; batch-1 prediction median / p95: 10.47 / 12.34 ms.

## Practical choice and benchmark method

Small is the strongest measured accuracy/latency tradeoff: comparable validation pose mAP and strict PCK versus medium, with 1.71× faster median prediction and 10.75M versus 23.55M parameters. Nano is the smallest checkpoint and has the lowest measured median latency, but its strict validation PCK is 1.72 percentage points lower than medium; its 0.42 ms latency advantage over small is a small difference in this single benchmark. No latency significance or deployment-camera guarantee is claimed.

Latency uses the same 100 validation images on RTX 5060 at 960 pixels, batch 1, default prediction precision, 20 warmup predictions per model, CUDA synchronization and preloaded images. Timings include preprocessing and postprocessing, exclude disk I/O, and are sequential single-pass measurements. GPU sharing, clocks, image shapes and deployment backend can affect results. Checkpoint sizes exclude optimizer state; small’s original optimizer state was preserved in best-training-state.pt and last.pt.
