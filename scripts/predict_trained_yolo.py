"""Inspect one validation example with the trained five-class model."""
import argparse,json,time
from pathlib import Path
from PIL import Image,ImageDraw
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
NAMES=['rice','noodles','dumplings','pizza','salad']
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--category',choices=NAMES,default='noodles')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--conf',type=float,default=.25)
    a=p.parse_args()
    if not 0<=a.conf<=1: p.error('conf must be between 0 and 1')
    data=ROOT/'data/uecfood100_detection'; candidates=[]
    for f in sorted((data/'labels/val').glob('*.txt')):
        if NAMES.index(a.category) in {int(x.split()[0]) for x in f.read_text().splitlines() if x.strip()}: candidates.append(f)
    if not 0<=a.index<len(candidates): p.error(f'index must be 0..{len(candidates)-1}')
    label=candidates[a.index]; image_path=data/'images/val'/(label.stem+'.jpg')
    with Image.open(image_path) as source: im=source.convert('RGB')
    model=YOLO(str(ROOT/'models/yolo_runs/baseline_10epochs/weights/best.pt'))
    if list(model.names.values())!=NAMES: raise ValueError('Unexpected model class mapping')
    start=time.perf_counter(); result=model.predict(im,device='cpu',imgsz=320,conf=a.conf,verbose=False)[0]; elapsed=time.perf_counter()-start
    truth=im.copy(); draw=ImageDraw.Draw(truth); w,h=im.size; ground=[]
    for line in label.read_text().splitlines():
        c,x,y,bw,bh=map(float,line.split()); box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        draw.rectangle(box,outline='lime',width=3); draw.text((box[0],box[1]),NAMES[int(c)],fill='red'); ground.append({'label':NAMES[int(c)],'xyxy':box})
    detections=[{'label':model.names[int(b.cls.item())],'confidence':float(b.conf.item()),'xyxy':b.xyxy[0].tolist()} for b in result.boxes]
    out=ROOT/'reports/validation_examples'/f'{a.category}_{a.index}_conf{a.conf:g}'; out.mkdir(parents=True,exist_ok=True)
    truth.save(out/'ground_truth.jpg'); Image.fromarray(result.plot()[:,:,::-1]).save(out/'prediction.jpg')
    report={'image':image_path.relative_to(ROOT).as_posix(),'split':'val','conf':a.conf,'predict_seconds_including_first_call_warmup':elapsed,'ground_truth':ground,'predictions':detections}
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); print('Output:',out)
if __name__=='__main__': main()
