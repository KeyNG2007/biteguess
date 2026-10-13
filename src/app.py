import asyncio, base64, io, json, os, threading, time
from collections import Counter
from pathlib import Path
import httpx
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
NAMES=['rice','noodles','dumplings','pizza','salad']
for line in (ROOT/'.env').read_text(encoding='utf-8-sig').splitlines() if (ROOT/'.env').exists() else []:
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip(chr(34)).strip(chr(39)))
app=FastAPI(title='美食猜猜看')
model=None
lock=threading.Lock()
def jpeg(image):
    b=io.BytesIO(); image.save(b,format='JPEG'); return b.getvalue()
def local_predict(image):
    global model
    start=time.perf_counter()
    with lock:
        if model is None: model=YOLO(str(ROOT/'models/yolo_runs/baseline_10epochs/weights/best.pt'))
        if list(model.names.values())!=NAMES: raise RuntimeError('模型类别配置不匹配')
        result=model.predict(image,device='cpu',imgsz=320,conf=.25,verbose=False)[0]
        boxes=[{'label':model.names[int(b.cls.item())],'confidence':round(float(b.conf.item()),4),'xyxy':b.xyxy[0].tolist()} for b in result.boxes]
        annotated=Image.fromarray(result.plot()[:,:,::-1])
    return {'status':'ok','model':'YOLO11n 五类微调','seconds':round(time.perf_counter()-start,3),'detections':boxes,'counts':dict(Counter(b['label'] for b in boxes)),'image':'data:image/jpeg;base64,'+base64.b64encode(jpeg(annotated)).decode(),'note':'耗时包含首次模型加载及排队；无框表示未检出，不等于确认未知类别。'}
async def remote_predict(raw):
    key=os.getenv('DEEPSEEK_API_KEY') or os.getenv('VISION_API_KEY')
    if not key: raise RuntimeError('尚未配置 API 密钥')
    base=os.getenv('DEEPSEEK_BASE_URL') or os.getenv('VISION_API_BASE_URL') or 'https://api.deepseek.com'
    name=os.getenv('DEEPSEEK_MODEL') or os.getenv('VISION_MODEL') or 'deepseek-flash'
    prompt='独立观察原图。只输出 JSON：{"foods":[{"label":"rice","count":1}],"unknown_present":false,"uncertain":false,"description":"简短中文说明"}。允许类别只有 rice 米饭料理、noodles 面食、dumplings 饺子、pizza 披萨、salad 沙拉（土豆、绿色、通心粉沙拉）。count 表示食物区域或份数，不是单只饺子数量。未知食物不要强行归类，设置 unknown_present；无法辨认设置 uncertain。不要给置信度或坐标。'
    payload={'model':name,'thinking':{'type':'disabled'},'max_tokens':1000,'response_format':{'type':'json_object'},'messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}}]}]}
    start=time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        response=await client.post(base.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload)
    if response.status_code!=200: raise RuntimeError(f'API 请求失败（HTTP {response.status_code}）')
    obj=response.json()
    if obj['choices'][0].get('finish_reason')!='stop': raise RuntimeError('API 输出未完整结束')
    answer=json.loads(obj['choices'][0]['message']['content'])
    if not isinstance(answer,dict) or not isinstance(answer.get('foods'),list): raise RuntimeError('API 返回格式不符合约定')
    for item in answer['foods']:
        if not isinstance(item,dict) or item.get('label') not in NAMES or type(item.get('count')) is not int or not 1<=item['count']<=100: raise RuntimeError('API 类别或数量不符合约定')
    if any(type(answer.get(k)) is not bool for k in ['unknown_present','uncertain']) or not isinstance(answer.get('description'),str): raise RuntimeError('API 状态字段不符合约定')
    return {'status':'ok','model':obj.get('model',name),'seconds':round(time.perf_counter()-start,3),'result':answer,'usage':obj.get('usage')}
async def safe_call(fn,*args):
    try: return await fn(*args)
    except httpx.TimeoutException: return {'status':'error','error':'大模型请求超时，请稍后重试'}
    except Exception as exc:
        message=str(exc) if isinstance(exc,RuntimeError) else '处理失败，请检查服务配置或图片'
        return {'status':'error','error':message}
async def local_async(image): return await asyncio.to_thread(local_predict,image)
@app.get('/',response_class=HTMLResponse)
def index(): return (ROOT/'src/index.html').read_text(encoding='utf-8')
@app.get('/health')
def health(): return {'status':'ok','weights_present':(ROOT/'models/yolo_runs/baseline_10epochs/weights/best.pt').is_file()}
@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...)):
    raw=await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024: raise HTTPException(413,'图片不能超过 10 MB')
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width*source.height>25_000_000: raise HTTPException(400,'图片像素过大')
            image=source.convert('RGB'); image.thumbnail((1024,1024))
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError): raise HTTPException(400,'无法读取图片，请使用 JPEG、PNG 或 WebP 图片')
    processed=jpeg(image)
    yolo,vision=await asyncio.gather(safe_call(local_async,image),safe_call(remote_predict,processed))
    return {'yolo':yolo,'vision':vision,'input_size':image.size,'scope':'两种方案独立读取同一预处理原图，数量为食物区域数。'}
