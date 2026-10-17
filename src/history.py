"""Single-host SQLite history; images and results share one transaction."""
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from sqlalchemy import create_engine, String, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

data_dir = Path(os.getenv('APP_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'work/storage')))
data_dir.mkdir(parents=True, exist_ok=True)
engine = create_engine('sqlite:///' + str(data_dir / 'history.db'), connect_args={'check_same_thread': False, 'timeout': 30})
class Base(DeclarativeBase): pass
class Record(Base):
    __tablename__ = 'analyses'
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String)
    note: Mapped[str] = mapped_column(Text, default='')
    image: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text)
Base.metadata.create_all(engine)

def save(image, result):
    import json
    record = Record(id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), image=image, result=json.dumps(result, ensure_ascii=False), note='')
    with Session(engine) as session:
        session.add(record)
        session.commit()
        return record.id

def serialize(record, detail=False):
    import json
    result = json.loads(record.result)
    item = {'id': record.id, 'created_at': record.created_at, 'note': record.note,
            'status': 'ok' if all(result[k]['status'] == 'ok' for k in ('yolo','vision')) else 'partial_or_failed'}
    if detail: item.update(image=record.image, result=result)
    return item

def listing(limit, offset):
    with Session(engine) as session:
        return [serialize(r) for r in session.scalars(select(Record).order_by(Record.created_at.desc()).offset(offset).limit(limit))]

def get(record_id):
    with Session(engine) as session:
        record = session.get(Record, record_id)
        return serialize(record, True) if record else None

def modify(record_id, note=None, delete=False):
    with Session(engine) as session:
        record = session.get(Record, record_id)
        if record is None: return False
        if delete: session.delete(record)
        else: record.note = note
        session.commit()
        return True
