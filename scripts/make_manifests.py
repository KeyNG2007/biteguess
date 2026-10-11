import csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/classes.json').read_text(encoding='utf-8'))
SEED = CFG['seed']
KNOWN = {item for values in CFG['groups'].values() for item in values}
UNKNOWN = {'sushi', 'ice_cream', 'steak'}
LABELS = {item: group for group, values in CFG['groups'].items() for item in values}
BASE = ROOT / 'data' / 'raw' / 'food-101'
OUT = ROOT / 'data' / 'manifests'
OUT.mkdir(parents=True, exist_ok=True)


def official(split, category):
    lines = (BASE / 'meta' / f'{split}.txt').read_text(encoding='utf-8').splitlines()
    return [x for x in lines if x.split('/', 1)[0] == category]


def rows(paths, source_split, limit=None, label_source=True):
    result = []
    for rel in paths[:limit] if limit else paths:
        category = rel.split('/', 1)[0]
        result.append({
            'path': str(Path('data/raw/food-101/images') / f'{rel}.jpg'),
            'original_category': category,
            'label': LABELS.get(category) if label_source else None,
            'source_split': source_split,
        })
    return result

rng = random.Random(SEED)
train, val = [], []
for category in sorted(KNOWN):
    items = official('train', category)
    rng.shuffle(items)
    cut = int(len(items) * 0.8)
    train.extend(rows(items[:cut], 'train'))
    val.extend(rows(items[cut:], 'train'))
known_test = []
for category in sorted(KNOWN):
    known_test.extend(rows(official('test', category), 'test', limit=20))
unknown_test = []
for category in sorted(UNKNOWN):
    unknown_test.extend(rows(official('test', category), 'test', limit=20, label_source=False))

for name, data in [('train', train), ('val', val), ('known_test', known_test), ('unknown_test', unknown_test)]:
    rng.shuffle(data)
    with (OUT / f'{name}.jsonl').open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

summary = {
    'seed': SEED,
    'known_categories': sorted(KNOWN),
    'unknown_categories': sorted(UNKNOWN),
    'counts': {name: len(data) for name, data in [('train', train), ('val', val), ('known_test', known_test), ('unknown_test', unknown_test)]},
    'label_counts': {group: sum(1 for x in train if x['label'] == group) for group in CFG['groups']},
}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
