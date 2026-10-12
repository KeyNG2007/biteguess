"""First full five-class detection baseline; test set is held out."""
import json
import time
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def main():
    model=YOLO(str(ROOT/'models/yolo11n.pt'))
    start=time.perf_counter()
    result=model.train(data=str(ROOT/'data/uecfood100_detection/data.yaml'),epochs=10,imgsz=320,batch=4,device='cpu',workers=0,seed=42,deterministic=True,project=str(ROOT/'models/yolo_runs'),name='baseline_10epochs',exist_ok=False,plots=True,cache=False,amp=False)
    report={'purpose':'first full training baseline; validation only, test held out','wall_seconds':time.perf_counter()-start,'save_dir':str(model.trainer.save_dir),'settings':{'epochs':10,'imgsz':320,'batch':4,'device':'cpu','seed':42},'metrics':result.results_dict}
    (ROOT/'reports/yolo_baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__': main()
