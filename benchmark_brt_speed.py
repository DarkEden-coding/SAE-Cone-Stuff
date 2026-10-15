import subprocess,json,time
from pathlib import Path
root=Path(__file__).resolve().parent
work=root/'datasets/brt-cone-pose'
results=[]
for name,flags in [('baseline',['--batch','4','--workers','6']),('workers10',['--batch','4','--workers','10']),('batch5',['--batch','5','--workers','6']),('compiled',['--batch','4','--workers','6','--compile']),('baseline-confirm',['--batch','4','--workers','6'])]:
    started=time.time()
    with (work/f'speed-{name}.log').open('w') as log:
        try:
            p=subprocess.run([str(root/'.venv/bin/python'),'-u',str(root/'train_brt_full.py'),'--benchmark',*flags],stdout=log,stderr=subprocess.STDOUT,timeout=420)
            code=p.returncode
        except subprocess.TimeoutExpired: code=124
    lines=(work/f'speed-{name}.log').read_text().splitlines()
    metrics=None
    for line in reversed(lines):
        try:
            obj=json.loads(line)
            if 'steady_images_per_second' in obj: metrics=obj;break
        except ValueError:pass
    results.append(dict(name=name,returncode=code,seconds=time.time()-started,metrics=metrics))
    (work/'speed-benchmarks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
