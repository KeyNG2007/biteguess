"""Reproducible five-class smoke training; never reads test labels."""
import json, random, time
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=ROOT/'data/uecfood100_detection'
    out=ROOT/'data/yolo_smoke'; out.mkdir(exist_ok=True)
    summary={}
    for split,quota in [('train',8),('val',3)]:
        pool={c:[] for c in range(5)}
        for label in sorted((data/'labels'/split).glob('*.txt')):
            classes={int(row.split()[0]) for row in label.read_text().splitlines() if row.strip()}
            for c in classes: pool[c].append(label.stem)
        selected=set(); rng=random.Random(42)
        for c,items in pool.items():
            if len(items)<quota: raise ValueError(f'Insufficient samples: {split}/{c}')
            selected.update(rng.sample(items,quota))
        paths=[(data/'images'/split/(s+'.jpg')).as_posix() for s in sorted(selected)]
        (out/(split+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
        counts={c:0 for c in range(5)}
        for s in selected:
            for row in (data/'labels'/split/(s+'.txt')).read_text().splitlines(): counts[int(row.split()[0])]+=1
        summary[split]={'images':len(paths),'boxes_by_class':counts}
    yaml=f'path: {out.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: rice\n  1: noodles\n  2: dumplings\n  3: pizza\n  4: salad\n'
    (out/'data.yaml').write_text(yaml,encoding='utf-8')
    print(json.dumps(summary),flush=True)
    start=time.perf_counter()
    model=YOLO(str(ROOT/'models/yolo11n.pt'))
    result=model.train(data=str(out/'data.yaml'),epochs=1,imgsz=320,batch=4,device='cpu',workers=0,seed=42,deterministic=True,project=str(ROOT/'models/yolo_runs'),name='smoke',exist_ok=False,plots=False,cache=False,amp=False)
    report={'purpose':'pipeline and timing only, not final accuracy','samples':summary,'wall_seconds':time.perf_counter()-start,'save_dir':str(model.trainer.save_dir),'settings':{'epochs':1,'imgsz':320,'batch':4,'device':'cpu','seed':42},'metrics':result.results_dict}
    (ROOT/'reports/yolo_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__': main()
