import json, tarfile, shutil
from pathlib import Path, PurePosixPath
root = Path(__file__).resolve().parents[1]
config = json.loads((root/'configs/classes.json').read_text(encoding='utf-8'))
selected = {x for group in config['groups'].values() for x in group} | {'sushi','ice_cream','steak'}
counts = dict.fromkeys(sorted(selected),0)
with tarfile.open(root/'data/food-101 (2).tar.gz','r|gz') as archive:
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if not member.isfile() or len(parts)<3 or parts[0]!='food-101':
            continue
        if not (parts[1]=='meta' or (parts[1]=='images' and parts[2] in selected)):
            continue
        if '..' in parts or PurePosixPath(member.name).is_absolute():
            raise ValueError('Unsafe path')
        target = root.joinpath('data','raw',*parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        with archive.extractfile(member) as source, target.open('wb') as output:
            shutil.copyfileobj(source,output)
        if target.stat().st_size != member.size:
            raise ValueError('Incomplete member')
        if parts[1]=='images':
            counts[parts[2]]+=1
            if sum(counts.values())%2000==0:
                print('Extracted',sum(counts.values()),flush=True)
base=root/'data/raw/food-101'
for category,count in counts.items():
    if count!=1000:
        raise ValueError((category,count))
for split,expected in [('train',750),('test',250)]:
    lines=(base/'meta'/f'{split}.txt').read_text().splitlines()
    for category in selected:
        entries=[x for x in lines if x.startswith(category+'/')]
        if len(entries)!=expected or any(not (base/'images'/(x+'.jpg')).is_file() for x in entries):
            raise ValueError((split,category))
(root/'reports/data_check.json').write_text(json.dumps({'counts':counts,'official_split_checked':True},indent=2),encoding='utf-8')
print('Selected categories and official splits verified.',flush=True)
