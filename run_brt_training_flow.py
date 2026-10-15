"""Wait for augmentation, verify its manifest, then train and evaluate unattended."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'datasets/brt-cone-pose'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--augmentation-pid',type=int,required=True)
    args=parser.parse_args()
    with (WORK/'training-flow.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A training flow is already running')
        manifest=WORK/'augmented-30k/train.txt'
        provenance=WORK/'augmented-30k/provenance.json'
        while not (manifest.exists() and provenance.exists()):
            try:os.kill(args.augmentation_pid,0)
            except ProcessLookupError:raise RuntimeError('Augmentation stopped before its manifest was complete')
            time.sleep(10)
        paths=[Path(line) for line in manifest.read_text().splitlines() if line.strip()]
        assert len(paths)==30000 and len(set(paths))==30000
        backgrounds=objects=0
        for image in paths:
            assert image.is_file(),image
            label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
            rows=np.asarray([list(map(float,line.split())) for line in label.read_text().splitlines()
                             if line.strip()]).reshape(-1,29)
            assert np.isfinite(rows).all(),label
            assert ((rows[:,1:5]>=0)&(rows[:,1:5]<=1)).all(),label
            kp=rows[:,5:].reshape(-1,8,3)
            assert ((kp[:,:,:2]>=0)&(kp[:,:,:2]<=1)).all(),label
            assert np.isin(kp[:,:,2],[0,1,2]).all(),label
            backgrounds+=len(rows)==0;objects+=len(rows)
        report={'training_images':len(paths),'background_images':backgrounds,'objects':objects,
                'augmented_images':len(json.loads(provenance.read_text())),
                'label_validation':'passed','validation_and_test':'original untouched splits'}
        (WORK/'augmentation-audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        env=os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        with (WORK/'full-training.log').open('a') as log:
            completed=subprocess.run(['systemd-inhibit','--what=sleep',
                                      '--who=BRT cone pose training',
                                      '--why=GPU training and evaluation in progress','--mode=block',
                                      sys.executable,'-u',str(ROOT/'train_brt_full.py'),
                                      '--batch','4','--epochs','20'],cwd=ROOT,env=env,
                                     stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Training or evaluation exited with status {completed.returncode}; inspect full-training.log')
        print('Training and held-out evaluation completed',flush=True)


if __name__=='__main__':main()
