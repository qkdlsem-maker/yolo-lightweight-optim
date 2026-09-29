"""Create an isolated source-verified training/validation view; never edit originals."""
import hashlib,json,zipfile,os
from pathlib import Path
R=Path('review_20260929_clean');R.mkdir(exist_ok=True)
data=R/'clean_data'
missing=json.loads((R/'missing137_names.json').read_text())
manifest={'phase':'new clean-label replication','excluded_unverified_images':missing,'splits':{}}
for split,expected_images,expected_boxes in [('train',69863,1286871),('val',10000,185526)]:
    archive=R/f'legacy_{split}_reconstructed.zip'
    out=data/'labels'/split;images=data/'images'/split
    out.mkdir(parents=True,exist_ok=True);images.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();names=[];boxes=0
    with zipfile.ZipFile(archive) as z:
        members=sorted(z.namelist());assert len(members)==expected_images
        for name in members:
            assert Path(name).name==name and name.endswith('.txt')
            stem=Path(name).stem;assert stem not in missing
            raw=z.read(name);lines=raw.decode().splitlines()
            for line in lines:
                if not line.strip():continue
                c,x,y,w,height=line.split();assert 0<=int(c)<10 and float(w)>0 and float(height)>0
                boxes+=1
            dest=out/name
            if dest.exists():assert dest.read_bytes()==raw
            else:dest.write_bytes(raw)
            original=(Path('data/raw')/split/'images'/(stem+'.jpg')).resolve();assert original.is_file(),original
            link=images/(stem+'.jpg')
            if link.is_symlink():assert link.resolve()==original
            else:assert not link.exists();os.symlink(original,link)
            h.update(name.encode()+b'\0'+raw);names.append(stem)
    assert boxes==expected_boxes
    (R/f'clean_{split}_image_ids.txt').write_text('\n'.join(names)+'\n')
    manifest['splits'][split]={'images':len(names),'boxes':boxes,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'ordered_label_content_sha256':h.hexdigest(),'image_id_manifest_sha256':hashlib.sha256((R/f'clean_{split}_image_ids.txt').read_bytes()).hexdigest()}
assert not set((R/'clean_train_image_ids.txt').read_text().splitlines())&set((R/'clean_val_image_ids.txt').read_text().splitlines())
names=['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train']
cfg='path: '+str(data.resolve())+'\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names))
p=R/'clean_bdd.yaml'
if p.exists():assert p.read_text()==cfg
else:p.write_text(cfg)
manifest['data_yaml_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
manifest['source_train_json_sha256']='c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e'
manifest['source_val_json_sha256']='c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af'
(R/'clean_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'complete','splits':manifest['splits'],'excluded':len(missing)},indent=2))
