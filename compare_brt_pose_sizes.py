"""Train nano/small sequentially and compare their held-out results to medium."""
import argparse
import csv
import fcntl
import gc
import json
from pathlib import Path
import shutil
import time

import torch
import yaml
from ultralytics import YOLO

from evaluate_brt_pose import evaluate
from train_brt_full import KeypointTrainer

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'datasets/brt-cone-pose/size-comparison'
MEDIUM = ROOT / 'runs/brt-pose-m-960-30k'
DATA = ROOT / 'datasets/brt-cone-pose/augmented-30k/data.yaml'
POSE = 'metrics/mAP50-95(P)'
BOX = 'metrics/mAP50-95(B)'


def write_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    tmp.replace(path)


def rows(run):
    with (run / 'results.csv').open() as handle:
        return [{k.strip(): float(v) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def summarize(run):
    history = rows(run)
    best = max(history, key=lambda row: row[POSE])
    budget20 = max((r for r in history if r['epoch'] <= 20), key=lambda r: r[POSE])
    return {'run': str(run), 'completed_epochs': int(history[-1]['epoch']),
            'best_validation_epoch': int(best['epoch']), 'best_validation': best,
            'best_validation_first_20_epochs': budget20,
            'validation_at_epoch20': next((r for r in history if r['epoch'] == 20), None),
            'strict_validation_through_20': json.loads((run / 'validation-pose-accuracy-through-20.json').read_text())
                if (run / 'validation-pose-accuracy-through-20.json').exists() else None,
            'validation_pose_accuracy': json.loads((run / 'validation-pose-accuracy.json').read_text())
                if (run / 'validation-pose-accuracy.json').exists() else None,
            'test_metrics': json.loads((run / 'test-metrics.json').read_text())
                if (run / 'test-metrics.json').exists() else None,
            'test_pose_accuracy': json.loads((run / 'pose-accuracy.json').read_text())
                if (run / 'pose-accuracy.json').exists() else None}


def category(delta):
    # Descriptive bands, not confidence intervals or significance tests.
    if delta > 0.01:
        return 'higher accuracy'
    if delta >= -0.01:
        return 'comparable accuracy (within 1 percentage point)'
    if delta >= -0.03:
        return 'moderately lower accuracy (1–3 percentage points)'
    return 'substantially lower accuracy (more than 3 percentage points)'


def report():
    results = {'medium': summarize(MEDIUM)}
    for size, title in [('n', 'nano'), ('s', 'small')]:
        run = ROOT / f'runs/brt-pose-{size}-960-30k-40'
        if (run / 'results.csv').exists():
            results[title] = summarize(run)
    baseline = results['medium']
    lines = ['# YOLO26 cone keypoints: model size comparison', '',
             'Medium trained for 20 epochs; nano and small have a 40-epoch cap. '
             'Same 30,000-image train manifest, untouched validation/test splits, 960-pixel input, '
             'batch 4, seed 26, AdamW and augmentation settings. '
             'Medium was resumed after epoch 2 with its schedule changed from 180 to 20; '
             'the new runs use a 40-epoch schedule from the start. '
             'The epoch-20 comparison controls epoch count, but not learning-rate schedule or mosaic closing.', '',
             'Checkpoint selection uses validation pose mAP50–95. Strict end-to-end PCK includes '
             'missed cones and low-confidence points. Categories are descriptive: within 1 percentage '
             'point of medium is comparable, 1–3 points lower is moderately lower, and more than '
             '3 points lower is substantially lower. Above 1 point is higher. '
             'These are single-seed results, not statistical significance claims.', '']
    if (WORK / 'early-stop.json').exists():
        stopped = json.loads((WORK / 'early-stop.json').read_text())
        lines += [f"Small was stopped by user request after {stopped['completed_epochs']} completed epochs. "
                  'The unfinished next epoch was discarded. Evaluation uses the saved best validation checkpoint; '
                  'its 40-epoch learning-rate schedule was not shortened retroactively.', '']
    for title, item in results.items():
        best = item['best_validation']
        lines += [f'## {title.title()}', '',
                  f"Completed epochs: {item['completed_epochs']}; best epoch: {item['best_validation_epoch']}.",
                  f"Best validation pose mAP50–95: {best[POSE]*100:.3f}%; box mAP50–95: {best[BOX]*100:.3f}%.",
                  f"Best validation pose mAP through epoch 20: {item['best_validation_first_20_epochs'][POSE]*100:.3f}%."]
        strict = item['validation_pose_accuracy']
        if strict:
            lines += [f"Validation end-to-end PCK at 5% / 10% box height: "
                      f"{strict['end_to_end_PCK_05']*100:.2f}% / {strict['end_to_end_PCK_10']*100:.2f}%."]
        if title != 'medium':
            delta20 = item['best_validation_first_20_epochs'][POSE] - baseline['best_validation'][POSE]
            lines += [f'Epoch-budget-20 pose mAP category: {category(delta20)} ({delta20*100:+.3f} percentage points).']
            strict20 = item['strict_validation_through_20']
            if strict20:
                lines += [f"Best-through-20 validation PCK at 5% / 10%: {strict20['end_to_end_PCK_05']*100:.2f}% / "
                          f"{strict20['end_to_end_PCK_10']*100:.2f}%."]
            delta = best[POSE] - baseline['best_validation'][POSE]
            item['validation_pose_map_category'] = category(delta)
            lines += [f'Pose mAP category versus medium: {category(delta)} ({delta*100:+.3f} percentage points).']
            if strict:
                delta = strict['end_to_end_PCK_05'] - baseline['validation_pose_accuracy']['end_to_end_PCK_05']
                item['strict_keypoint_category'] = category(delta)
                lines += [f'Strict keypoint category versus medium: {category(delta)} ({delta*100:+.2f} percentage points).']
        if item['test_metrics']:
            lines += [f"Test pose / box mAP50–95: {item['test_metrics'][POSE]*100:.3f}% / {item['test_metrics'][BOX]*100:.3f}%."]
        if item['test_pose_accuracy']:
            strict_test = item['test_pose_accuracy']
            lines += [f"Test end-to-end PCK at 5% / 10%: {strict_test['end_to_end_PCK_05']*100:.2f}% / "
                      f"{strict_test['end_to_end_PCK_10']*100:.2f}%."]
        speed_file = Path(item['run']) / 'comparison-inference.json'
        if speed_file.exists():
            speed = json.loads(speed_file.read_text())
            item['inference'] = speed
            lines += [f"Parameters: {speed['parameters']:,}; checkpoint: {speed['checkpoint_mib']:.2f} MiB; "
                      f"batch-1 prediction median / p95: {speed['median_ms']:.2f} / {speed['p95_ms']:.2f} ms."]
        lines += ['']
    write_json(WORK / 'comparison.json', results)
    (WORK / 'COMPARISON.md').write_text('\n'.join(lines))


def benchmark_inference():
    import numpy as np
    import cv2
    config = yaml.safe_load(DATA.read_text())
    images = sorted(Path(config['val']).glob('*.jpg'))
    # Identical original images, resolution, batch size and thresholds for all models.
    indices = np.linspace(0, len(images)-1, 100).astype(int)
    samples = [cv2.imread(str(images[i])) for i in indices]
    for run in (MEDIUM, ROOT/'runs/brt-pose-n-960-30k-40', ROOT/'runs/brt-pose-s-960-30k-40'):
        status('benchmarking', run=str(run))
        model = YOLO(str(run/'weights/best.pt'))
        parameters = sum(p.numel() for p in model.model.parameters())
        durations = []
        for i, sample in enumerate(samples[:20] + samples):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(sample, imgsz=960, device=0, conf=0.25, verbose=False)
            torch.cuda.synchronize()
            if i >= 20:
                durations.append((time.perf_counter()-start)*1000)
        write_json(run/'comparison-inference.json', {
            'parameters': parameters, 'checkpoint_mib': (run/'weights/best.pt').stat().st_size/2**20,
            'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations,95)),
            'gpu': torch.cuda.get_device_name(0), 'images': 100, 'warmup_images': 20,
            'method': 'batch-1 predict wall time, CUDA synchronized, images preloaded; includes preprocessing/postprocessing; default prediction precision',
            'imgsz': 960})
        del model
        gc.collect()
        torch.cuda.empty_cache()


def status(stage, **kwargs):
    write_json(WORK / 'status.json', {'stage': stage,
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs})


def train(size):
    name = f'brt-pose-{size}-960-30k-40'
    run = ROOT / 'runs' / name
    if run.exists():
        raise RuntimeError(f'Refusing to overwrite existing run: {run}')
    settings = yaml.safe_load((MEDIUM / 'args.yaml').read_text())
    # Retain all saved training settings except run identity and budget.
    for key in ('save_dir', 'model', 'resume'):
        settings.pop(key, None)
    settings.update(data=str(DATA), epochs=40, patience=30, batch=4,
                    project=str(ROOT / 'runs'), name=name, exist_ok=False,
                    resume=False, plots=False)
    model = YOLO(f'yolo26{size}-pose.pt')

    def on_epoch(trainer):
        epoch = trainer.epoch + 1
        status('training', size=size, epoch=epoch, epochs=40,
               run=str(run), metrics=trainer.metrics)
        # best.pt is saved before this callback. Freeze the best-so-far at epoch 20.
        if epoch == 20:
            shutil.copy2(run / 'weights/best.pt', run / 'weights/best-through-20.pt')

    model.add_callback('on_fit_epoch_end', on_epoch)
    status('starting', size=size, epochs=40, run=str(run))
    model.train(trainer=KeypointTrainer, **settings)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    evaluate_run(size)


def evaluate_run(size):
    name = f'brt-pose-{size}-960-30k-40'
    run = ROOT / 'runs' / name
    status('evaluating', size=size, run=str(run))
    best = YOLO(str(run / 'weights/best.pt'))
    evaluate(best, DATA, run / 'validation-pose-accuracy.json', imgsz=960, split='val')
    metrics = best.val(data=str(DATA), split='test', device=0, batch=4,
                       imgsz=960, project=str(ROOT / 'runs'), name=name+'-test', plots=False)
    write_json(run / 'test-metrics.json', metrics.results_dict)
    evaluate(best, DATA, run / 'pose-accuracy.json', imgsz=960, split='test')
    del best
    gc.collect()
    torch.cuda.empty_cache()
    if (run / 'weights/best-through-20.pt').exists():
        best20 = YOLO(str(run / 'weights/best-through-20.pt'))
        evaluate(best20, DATA, run / 'validation-pose-accuracy-through-20.json', imgsz=960, split='val')
        del best20
        gc.collect()
        torch.cuda.empty_cache()
    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate existing checkpoints without training')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report()
        return
    with (WORK / 'flow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA GPU required')
            torch.set_num_threads(6)
            report()
            if args.eval_only:
                for size in ('n', 's'):
                    run = ROOT / f'runs/brt-pose-{size}-960-30k-40'
                    required = ('validation-pose-accuracy.json', 'test-metrics.json',
                                'pose-accuracy.json', 'validation-pose-accuracy-through-20.json')
                    if not all((run / path).exists() for path in required):
                        evaluate_run(size)
            else:
                for size in ('n', 's'):
                    train(size)
            benchmark_inference()
            report()
            status('complete', report=str(WORK / 'COMPARISON.md'))
        except Exception as exc:
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
