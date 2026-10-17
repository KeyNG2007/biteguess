"""Check real local inference and upload contracts; remote API is opt-in."""
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from PIL import Image
from src import app as service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    image = Image.open(args.image).convert('RGB')
    image.thumbnail((1024, 1024))
    local = service.local_predict(image)
    box = local['detections'][0]['xyxy'] if local['detections'] else None
    prompt = {'box': box} if box else {'point': [image.width / 2, image.height / 2]}
    sam = service.segment_local(image, prompt)
    report = {'yolo': {k: v for k, v in local.items() if k != 'image'},
              'sam': {k: v for k, v in sam.items() if k != 'image'},
              'remote': {'status': 'not_run'}}
    client = TestClient(service.app)
    assert client.get('/').status_code == 200
    assert client.get('/health').status_code == 200
    assert client.post('/api/analyze', files={'file': ('bad.jpg', b'bad', 'image/jpeg')}).status_code == 400
    assert client.post('/api/analyze', files={'file': ('big.jpg', b'x' * (10 * 1024 * 1024 + 1), 'image/jpeg')}).status_code == 413
    data = io.BytesIO()
    image.save(data, format='JPEG')
    assert client.post('/api/segment', files={'file': ('image.jpg', data.getvalue(), 'image/jpeg')}, data={'prompt': '{"point":[-1,0]}'}).status_code == 400
    original_remote = service.remote_predict
    async def unavailable(raw):
        raise RuntimeError('模拟外部服务不可用')
    try:
        service.remote_predict = unavailable
        response = client.post('/api/analyze', files={'file': ('image.jpg', data.getvalue(), 'image/jpeg')})
        assert response.status_code == 200
        assert response.json()['yolo']['status'] == 'ok'
        assert response.json()['vision']['status'] == 'error'
        report['partial_failure'] = 'passed (simulated remote failure, real YOLO)'
    finally:
        service.remote_predict = original_remote
    if args.remote:
        import asyncio
        remote = asyncio.run(service.safe_call(original_remote, data.getvalue()))
        report['remote'] = remote
    report['contracts'] = 'passed'
    report['readiness'] = client.get('/ready').json()
    destination = ROOT / 'work/stage3-baseline.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.remote and report['remote']['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
