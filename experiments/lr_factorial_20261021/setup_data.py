"""Read-only provenance checks and an isolated, deterministic smoke subset."""
import json,hashlib,time
from pathlib import Path
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
R=Path('review_20261021_lr_factorial');P=json.loads((R/'protocol.json').read_text());old=Path(P['previous_root'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(old/'clean_data_manifest.json')==P['data']['manifest_sha256']
assert sha(P['data']['yaml'])==P['data']['yaml_sha256']
m=json.loads((old/'clean_data_manifest.json').read_text())
for split in ['train','val']:
    h=hashlib.sha256();labels=sorted((old/'clean_data/labels'/split).glob('*.txt'))
    assert len(labels)==m['splits'][split]['images']
    for p in labels:h.update(p.name.encode()+b'\0'+p.read_bytes())
    assert h.hexdigest()==m['splits'][split]['ordered_label_content_sha256']
    images=sorted((old/'clean_data/images'/split).glob('*.jpg'))
    assert len(images)==len(labels) and all(p.is_file() for p in images)
    n=128 if split=='train' else 32
    selected=sorted(images,key=lambda p:hashlib.sha256(('preflight20261017:'+p.name).encode()).hexdigest())[:n]
    dest=R/f'smoke_{split}.txt';text='\n'.join(str(p.resolve()) for p in selected)+'\n'
    # Use the verified reconstructed view, retaining its /images/ -> /labels/ mapping.
    text='\n'.join(str(p.absolute()) for p in selected)+'\n'
    if dest.exists():assert dest.read_text()==text
    else:dest.write_text(text)
cfg=yaml.safe_load(Path(P['data']['yaml']).read_text())
smoke={'path':str(Path.cwd()),'train':str((R/'smoke_train.txt').absolute()),'val':str((R/'smoke_val.txt').absolute()),'names':cfg['names']}
s=yaml.safe_dump(smoke,sort_keys=False);dest=R/'smoke.yaml'
if dest.exists():assert dest.read_text()==s
else:dest.write_text(s)
d=check_det_dataset(P['data']['yaml']);hyp=get_cfg();hyp.imgsz=640;hyp.mosaic=0.
dataset=build_yolo_dataset(hyp,d['train'],64,d,mode='train',rect=False)
assert len(dataset)==P['data']['train_images']
boxes=sum(len(v['cls']) for v in dataset.labels);assert boxes==P['data']['native_training_boxes']
record={'status':'passed','train_images':len(dataset),'native_training_boxes':boxes,'source_manifest_sha256':sha(old/'clean_data_manifest.json'),
  'all_train_and_validation_label_hashes_match':True,'all_image_links_resolve':True,'smoke_yaml_sha256':sha(dest),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(R/'data_verification.json').write_text(json.dumps(record,indent=2)+'\n');print('DATA_VERIFIED',flush=True)
