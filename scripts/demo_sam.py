"""SAM point/box prompts and YOLO-to-SAM integration on a validation image."""
import io,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO,SAM
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=ROOT/'data/uecfood100_detection'; label=next(iter(sorted((data/'labels/val').glob('*.txt'))))
    source=data/'images/val'/(label.stem+'.jpg')
    image=Image.open(source).convert('RGB'); w,h=image.size
    c,x,y,bw,bh=map(float,label.read_text().splitlines()[0].split()); box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]; point=[x*w,y*h]
    yolo=YOLO(str(ROOT/'models/yolo_runs/baseline_10epochs/weights/best.pt'))
    detections=yolo.predict(image,device='cpu',imgsz=320,conf=.25,verbose=False)[0]
    sam=SAM(str(ROOT/'models/sam2.1_t.pt'))
    prompts={'point':{'points':[point],'labels':[1]},'ground_truth_box':{'bboxes':[box]}}
    if len(detections.boxes): prompts['yolo_box']={'bboxes':detections.boxes.xyxy.cpu().tolist()}
    out=ROOT/'reports/sam_demo'; out.mkdir(exist_ok=True); report={'image':source.relative_to(ROOT).as_posix(),'model':'sam2.1_t.pt','device':'cpu','point':point,'ground_truth_box':box,'experiments':{},'note':'Prompt demonstration only; dataset has boxes, not segmentation ground truth. No segmentation accuracy claimed.'}
    for name,prompt in prompts.items():
        start=time.perf_counter(); results=sam.predict(image,device='cpu',verbose=False,**prompt); elapsed=time.perf_counter()-start
        result=results[0]; n=len(result.masks.data) if result.masks is not None else 0
        Image.fromarray(result.plot()[:,:,::-1]).save(out/(name+'.jpg'))
        if n:
            masks=result.masks.data.cpu().numpy(); np.savez_compressed(out/(name+'_masks.npz'),masks=masks)
        report['experiments'][name]={'seconds':elapsed,'masks':n,'prompt':prompt}
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
