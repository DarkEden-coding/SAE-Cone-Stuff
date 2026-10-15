"""Fresh GPU pose training with checkpoint selection driven by keypoint accuracy."""
import argparse
import json
from pathlib import Path
import time

import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "datasets/brt-cone-pose"
DATA = WORK / "augmented-30k/data.yaml"
STATUS = WORK / "full-training-status.json"


class KeypointTrainer(PoseTrainer):
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            # Upstream restores epochs/compile from checkpoint; honor explicit settings.
            for key in ("epochs", "compile", "channels_last"):
                if key in overrides:
                    setattr(self.args, key, overrides[key])

    def validate(self):
        previous_best = self.best_fitness
        metrics, _ = super().validate()
        # Default fitness sums box and pose metrics. This task prioritizes pose.
        fitness = float(metrics["metrics/mAP50-95(P)"])
        self.best_fitness = fitness if previous_best is None else max(previous_best,fitness)
        return metrics, fitness


def status(stage, **kwargs):
    payload = {"stage":stage,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**kwargs}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,default=str)+"\n")
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--batch",type=float,default=0.60)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--compile",action="store_true")
    parser.add_argument("--channels-last",action="store_true",default=None)
    parser.add_argument("--resume",type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is required; refusing CPU training")
    torch.set_num_threads(6)
    config = yaml.safe_load(DATA.read_text())
    data = DATA
    if args.benchmark:
        directory = WORK / "benchmark"
        directory.mkdir(exist_ok=True)
        source = Path(config["path"])
        for split,count in (("train",256),("val",32)):
            images = sorted((source/"images"/split).glob("*.jpg"))
            # Spread across source teams instead of only the first alphabetic group.
            indices = torch.linspace(0,len(images)-1,count-8).long().tolist()
            label_root=source/"labels"/split
            crowded=sorted(range(len(images)),key=lambda i:len((label_root/f"{images[i].stem}.txt").read_text().splitlines()),reverse=True)
            indices=list(dict.fromkeys(indices+crowded))[:count]
            manifest=directory/f"{split}.txt"
            manifest.write_text("\n".join(str(images[i]) for i in indices)+"\n")
            config[split]=str(manifest)
        data=directory/"data.yaml"
        data.write_text(yaml.safe_dump(config,sort_keys=False))
    name = "brt-pose-m-960-benchmark" if args.benchmark else "brt-pose-m-960-30k"
    model = YOLO(str(args.resume or ROOT/"yolo26m-pose.pt"))
    batch = int(args.batch) if args.batch>=1 else args.batch

    def on_epoch(trainer):
        if args.benchmark:
            return
        status("training",epoch=trainer.epoch+1,epochs=trainer.epochs,
               batch=trainer.batch_size,save_dir=trainer.save_dir,
               metrics=trainer.metrics,best_pose_fitness=trainer.best_fitness,
               gpu=torch.cuda.get_device_name(0))

    benchmark_times=[]
    def benchmark_batch(trainer):
        if args.benchmark:
            torch.cuda.synchronize()
            benchmark_times.append(time.perf_counter())
    model.add_callback("on_train_batch_end",benchmark_batch)
    model.add_callback("on_fit_epoch_end",on_epoch)
    model.add_callback("on_train_start",lambda trainer:torch.cuda.reset_peak_memory_stats())
    if not args.benchmark:
        status("starting",epochs=args.epochs,batch=batch,gpu=torch.cuda.get_device_name(0))
    try:
        model.train(
            trainer=KeypointTrainer,data=str(data),device=0,
            epochs=1 if args.benchmark else args.epochs,patience=30,
            batch=batch,imgsz=960,workers=args.workers,cache=False,
            compile=args.compile,channels_last=args.channels_last,
            project=str(ROOT/"runs"),name=name,resume=bool(args.resume),
            optimizer="AdamW",lr0=0.0015,lrf=0.02,cos_lr=True,
            warmup_epochs=3,weight_decay=0.0005,
            fliplr=0,flipud=0,degrees=0,perspective=0,shear=0,
            translate=0.03,scale=0.15,mosaic=0 if args.benchmark else 0.15,
            close_mosaic=5,mixup=0,cutmix=0,hsv_h=0.005,hsv_s=0.25,hsv_v=0.25,
            seed=26,deterministic=True,amp=True,plots=False,save_period=10,
        )
        run=Path(model.trainer.save_dir)
        if args.benchmark:
            print(json.dumps({"batch":model.trainer.batch_size,
                              "steady_images_per_second":(len(benchmark_times)-17)*model.trainer.batch_size/(benchmark_times[-1]-benchmark_times[16]),
                              "workers":args.workers,"compile":args.compile,
                              "peak_cuda_gib":torch.cuda.max_memory_reserved()/2**30,
                              "run":str(run)}),flush=True)
            return
        status("evaluating",save_dir=run,best_checkpoint=run/"weights/best.pt")
        # Test data is untouched during training and checkpoint selection.
        best=YOLO(str(run/"weights/best.pt"))
        from evaluate_brt_pose import evaluate
        validation_accuracy=evaluate(best,DATA,run/"validation-pose-accuracy.json",imgsz=960,split="val")
        metrics=best.val(data=str(DATA),split="test",device=0,batch=model.trainer.batch_size,
                         imgsz=960,project=str(ROOT/"runs"),name="brt-pose-m-960-30k-test",plots=True)
        (run/"test-metrics.json").write_text(json.dumps(metrics.results_dict,indent=2))
        report=evaluate(best,DATA,run/"pose-accuracy.json",imgsz=960)
        status("complete",save_dir=run,best_checkpoint=run/"weights/best.pt",
               test_metrics=metrics.results_dict,pose_accuracy=report,validation_pose_accuracy=validation_accuracy)
    except Exception as exc:
        if not args.benchmark:
            status("failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
