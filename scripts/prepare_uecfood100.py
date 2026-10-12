"""Convert selected UECFOOD100 boxes; split by decoded image content."""
import argparse, hashlib, io, json, random, zipfile
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT/'data/dataset100.zip')
    args = parser.parse_args()
    config=json.loads((ROOT/'configs/detection_classes.json').read_text(encoding='utf-8'))
    names=list(config['groups']); mapping={cat:i for i,name in enumerate(names) for cat in config['groups'][name]}
    records={}; id_hash={}; duplicates=0
    with zipfile.ZipFile(args.archive) as z:
        for cat,cls in mapping.items():
            prefix=f'UECFOOD100/{cat}/'
            for line in z.read(prefix+'bb_info.txt').decode().splitlines()[1:]:
                if not line.strip(): continue
                parts=line.split(); ident=parts[0]; box=tuple(map(int,parts[1:]))
                if len(box)!=4: raise ValueError('Malformed annotation')
                member=prefix+ident+'.jpg'
                raw=z.read(member)
                with Image.open(io.BytesIO(raw)) as image:
                    rgb=image.convert('RGB'); w,h=rgb.size
                    digest=hashlib.sha256(str((w,h)).encode()+rgb.tobytes()).hexdigest()
                if ident in id_hash and id_hash[ident]!=digest:
                    raise ValueError(f'Conflicting image ID {ident}')
                id_hash[ident]=digest
                x1,y1,x2,y2=box
                if not (0<=x1<x2<=w and 0<=y1<y2<=h):
                    raise ValueError(f'Invalid box {member}: {box}')
                rec=records.setdefault(digest,{'member':member,'size':(w,h),'ids':set(),'boxes':set()})
                label=(cls,*box)
                if label in rec['boxes']: duplicates+=1
                rec['boxes'].add(label); rec['ids'].add(ident)
        # Stratify by all supported classes present; exact content duplicates stay together.
        strata={}
        for digest,rec in sorted(records.items()):
            key=tuple(sorted({b[0] for b in rec['boxes']}))
            strata.setdefault(key,[]).append(digest)
        rng=random.Random(config['seed']); splits={'train':[],'val':[],'test':[]}
        for key,items in sorted(strata.items()):
            rng.shuffle(items); n=len(items)
            nv=round(n*.15); nt=round(n*.15)
            splits['val'].extend(items[:nv]); splits['test'].extend(items[nv:nv+nt]); splits['train'].extend(items[nv+nt:])
        out=ROOT/'data/uecfood100_detection'
        if (out/'manifest.json').exists(): raise SystemExit('Output already exists; use a new output directory instead of overwriting.')
        stats={}; manifest=[]
        for split,items in splits.items():
            idir=out/'images'/split; ldir=out/'labels'/split
            idir.mkdir(parents=True,exist_ok=True); ldir.mkdir(parents=True,exist_ok=True)
            counts=Counter(); image_counts=Counter()
            for digest in sorted(items):
                rec=records[digest]; w,h=rec['size']; stem=digest
                (idir/(stem+'.jpg')).write_bytes(z.read(rec['member']))
                labels=[]
                for cls,x1,y1,x2,y2 in sorted(rec['boxes']):
                    labels.append(f'{cls} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                    counts[names[cls]]+=1
                for cls in {b[0] for b in rec['boxes']}: image_counts[names[cls]]+=1
                (ldir/(stem+'.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
                manifest.append({'split':split,'sha256_pixels':digest,'source_ids':sorted(rec['ids']),'source_member':rec['member'],'boxes':sorted(rec['boxes'])})
            stats[split]={'images':len(items),'boxes_by_class':dict(counts),'images_by_class':dict(image_counts)}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        yaml='path: '+out.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(names))
        (out/'data.yaml').write_text(yaml,encoding='utf-8')
        report={'seed':config['seed'],'split_method':'70/15/15 within supported-class-set strata, rounded; grouped by decoded pixel SHA256','source_unique_ids':len(id_hash),'unique_images':len(records),'duplicate_box_rows_removed':duplicates,'splits':stats,'scope':'counts refer to annotated food regions, not individual dumplings; exact duplicates grouped, near duplicates not automatically resolved','source':config['source'],'usage':config['usage']}
        (ROOT/'reports/detection_data_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        multi=sorted(records,key=lambda d:(-len({b[0] for b in records[d]['boxes']}),d))[:4]
        canvas=Image.new('RGB',(1000,600),'white')
        for i,digest in enumerate(multi):
            rec=records[digest]
            im=Image.open(io.BytesIO(z.read(rec['member']))).convert('RGB'); draw=ImageDraw.Draw(im)
            for cls,*box in sorted(rec['boxes']):
                draw.rectangle(box,outline='lime',width=3); draw.text((box[0],box[1]),names[cls],fill='red')
            im.thumbnail((490,290)); canvas.paste(im,((i%2)*500,(i//2)*300))
        canvas.save(out/'annotation_samples.jpg')
        print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
