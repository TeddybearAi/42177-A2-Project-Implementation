"""Extract only manifest-selected PlantVillage images, preserving team split and IDs."""
import argparse, csv, hashlib, json, zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
    rows=list(csv.DictReader(open(a.manifest)));rows=[r for r in rows if r['root']=='pv']
    assert len(rows)==17392
    selected=[];counts={}
    for r in rows:
        key=(r['split'],r['label']);counts[key]=counts.get(key,0)+1
        if a.full or (r['split'] in ('train','val') and counts[key] <= (4 if r['split']=='train' else 2)):
            selected.append(r)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(a.archive) as z:
        # Determine one archive prefix that contains the exact shared relative paths.
        candidates=[n[:-len(rows[0]['relpath'])] for n in z.namelist() if n.endswith('/'+rows[0]['relpath']) or n==rows[0]['relpath']]
        candidates=[prefix for prefix in candidates if all(prefix+r['relpath'] in z.NameToInfo for r in rows)]
        assert candidates, 'Archive cannot reproduce all team PV paths'
        prefix=sorted(candidates,key=lambda s:(len(s),s))[0]
        for r in selected:
            dest=(out/r['relpath']).resolve();assert dest.is_relative_to(out.resolve())
            b=z.read(prefix+r['relpath']);assert hashlib.md5(b).hexdigest()==r['md5'],r['image_id']
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    report={'all_pv_paths_found':len(rows),'extracted_md5_verified':len(selected),'archive_prefix':prefix,
            'mode':'full' if a.full else 'smoke','plantdoc_images_opened':0}
    (out/'extraction_audit.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
