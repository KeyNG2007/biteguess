"""Fixed small held-out comparison; do not tune using these results."""
import asyncio,json,random,time
from collections import Counter
from pathlib import Path
from PIL import Image
from src.app import NAMES,jpeg,local_predict,remote_predict
ROOT=Path(__file__).resolve().parents[1]
async def main():
    out=ROOT/'reports/comparison'; out.mkdir(exist_ok=True)
    if (out/'results.json').exists(): raise SystemExit('Existing evaluation retained; refusing to overwrite')
    data=ROOT/'data/uecfood100_detection'; rng=random.Random(42); selected=set()
    for c in range(5):
        candidates=[p for p in sorted((data/'labels/test').glob('*.txt')) if c in {int(row.split()[0]) for row in p.read_text().splitlines() if row.strip()}]
        selected.update(rng.sample(candidates,2))
    protocol={'seed':42,'split':'test','selection':'2 images containing each class; duplicates merged; may include other supported classes','images':[p.stem for p in sorted(selected)],'weights':'models/yolo_runs/baseline_10epochs/weights/best.pt','yolo_conf':.25,'yolo_imgsz':320,'max_image_edge':1024,'vision_model':'deepseek-flash','metrics':'exact supported-class set match and sum absolute region-count errors; API errors separate','limitations':'small sample; known foods only; unknown_present cannot be scored from selected-class annotations; no segmentation accuracy; not a reliable full-dataset estimate'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]
    for label in sorted(selected):
        truth=Counter(NAMES[int(row.split()[0])] for row in label.read_text().splitlines() if row.strip())
        with Image.open(data/'images/test'/(label.stem+'.jpg')) as im: image=im.convert('RGB'); image.thumbnail((1024,1024))
        local=await asyncio.to_thread(local_predict,image)
        try:
            remote=await remote_predict(jpeg(image))
            predicted=Counter()
            for f in remote['result']['foods']: predicted[f['label']]+=f['count']
        except Exception:
            remote={'status':'error','error':'API failed during fixed evaluation'}; predicted=None
        row={'image':label.stem,'truth':dict(truth),'yolo':{k:v for k,v in local.items() if k!='image'},'vision':remote}
        for key,counts in [('yolo',local['counts']),('vision',predicted)]:
            if counts is not None: row[key]['evaluation']={'class_set_correct':set(counts)==set(truth),'absolute_count_error':sum(abs(counts.get(n,0)-truth.get(n,0)) for n in NAMES)}
        rows.append(row); (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Completed',len(rows),'/',len(selected),flush=True)
    summary={'images':len(rows),'limitations':protocol['limitations']}
    for key in ['yolo','vision']:
        valid=[x[key] for x in rows if 'evaluation' in x[key]]
        summary[key]={'successful_images':len(valid),'errors':len(rows)-len(valid),'class_set_accuracy':sum(x['evaluation']['class_set_correct'] for x in valid)/len(valid) if valid else None,'mean_absolute_region_count_error':sum(x['evaluation']['absolute_count_error'] for x in valid)/len(valid) if valid else None,'mean_seconds':sum(x['seconds'] for x in valid)/len(valid) if valid else None}
    summary['vision_total_tokens']=sum((x['vision'].get('usage') or {}).get('total_tokens',0) for x in rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
if __name__=='__main__': asyncio.run(main())
