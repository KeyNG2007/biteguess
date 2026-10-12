"""One-image YOLO detection lesson; no training or dataset download."""
import argparse
import json
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/yolo11n.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'yolo_demo_conf025')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    image_path = args.image or next(iter(sorted((ROOT / 'data/raw/food-101/images/pizza').glob('*.jpg'))), None)
    if image_path is None or not image_path.is_file():
        raise SystemExit('No image found; supply --image PATH')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Pass a PIL image to avoid OpenCV path issues with Chinese folder names.
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    model = YOLO(str(args.model))
    start = time.perf_counter()
    result = model.predict(image, device='cpu', conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    elapsed = time.perf_counter() - start
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({'class_id': class_id, 'label': result.names[class_id], 'confidence': float(box.conf.item()), 'xyxy': box.xyxy[0].tolist()})
    report = {'image': str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path), 'model': str(args.model), 'imgsz': args.imgsz, 'device': 'cpu', 'conf_threshold': args.conf, 'predict_seconds': elapsed, 'timing_note': 'First prediction includes predictor warmup; model load and image read excluded.', 'detections': detections, 'class_names': result.names}
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    Image.fromarray(result.plot()[:, :, ::-1]).save(out / 'prediction.jpg')
    print(json.dumps({k:v for k,v in report.items() if k != 'class_names'}, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
