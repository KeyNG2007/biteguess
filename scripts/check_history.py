"""Verify persistence and CRUD against an isolated temporary database."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with tempfile.TemporaryDirectory(prefix='biteguess-history-') as directory:
    os.environ['APP_DATA_DIR'] = directory
    from fastapi.testclient import TestClient
    from src.app import app
    from src import history
    client = TestClient(app)
    record_id = history.save('data:image/jpeg;base64,test', {'yolo': {'status': 'ok'}, 'vision': {'status': 'error'}})
    path = '/api/history/' + record_id
    assert client.get('/api/history').json()[0]['id'] == record_id
    assert client.patch(path, json={'note': '测试备注'}).status_code == 200
    history.engine.dispose()
    assert client.get(path).json()['note'] == '测试备注'
    assert client.patch(path, json={'note': 'x' * 2001}).status_code == 422
    assert client.delete(path).status_code == 200
    assert client.get(path).status_code == 404
    assert client.get('/api/history?limit=101').status_code == 422
    assert client.get('/history').status_code == 200
    history.engine.dispose()
    print('PASS: CRUD, database reconnect persistence, validation, history page')
