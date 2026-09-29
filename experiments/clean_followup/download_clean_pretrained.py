import urllib.request,hashlib,json
from pathlib import Path
R=Path('review_20260929_clean/pretrained');R.mkdir(parents=True,exist_ok=True)
records=[]
for name in ['yolov8n.pt','yolov8l.pt']:
    p=R/name;url='https://github.com/ultralytics/assets/releases/download/v8.2.0/'+name
    if not p.exists():
        partial=p.with_suffix('.partial')
        with urllib.request.urlopen(url,timeout=60) as r,partial.open('wb') as f:
            while b:=r.read(1024*1024):f.write(b)
        partial.replace(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    import torch
    obj=torch.load(p,map_location='cpu',weights_only=False)
    data=str(obj.get('train_args',{}).get('data',''))
    assert 'coco' in data.lower(),(name,data)
    assert obj['model'].model[-1].nc==80
    records.append({'name':name,'url':url,'bytes':p.stat().st_size,'sha256':digest,'pretraining_data_argument':data,'classes':80})
    print('OFFICIAL_COCO_INPUT_VERIFIED',name,digest,flush=True)
(R/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
