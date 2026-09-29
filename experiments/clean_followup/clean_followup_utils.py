import hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def model_roster():
    root=Path('review_20260929_clean');old=Path('review_20260929');models=[]
    archive=[('nano','yolov8n_baseline'),('pruned_r15','yolov8n_pruned_r15_finetuned_v2'),('pruned_r30','yolov8n_pruned_finetuned'),('pruned_r45','yolov8n_pruned_r45_finetuned'),('recovery_kd','yolov8n_pruned_r45_kd')]
    for short,name in archive:models.append({'name':'archived_'+short,'checkpoint':str(Path('outputs')/name/'weights/best.pt'),'prior_hash_record':str(old/'stratified'/name/'metrics.json'),'group':'archived'})
    for arm in ['control','mse','cwd']:
        for seed in range(3):
            name=f'{arm}_seed{seed}_e20';models.append({'name':'earlier_'+name,'checkpoint':str(old/'confirmatory'/name/'ema_epoch20.pt'),'prior_hash_record':str(old/'stratified'/name/'metrics.json'),'group':'earlier_continuation','arm':arm,'seed':seed})
    for name in ['nano','large']:models.append({'name':'clean_'+name,'checkpoint':str(root/'base'/name/'weights/last.pt'),'prior_hash_record':str(root/'base_records'/name/'COMPLETE.json'),'group':'clean_base'})
    for arm in ['control','mse','cwd']:
        for seed in range(3):
            name=f'{arm}_seed{seed}_e20';models.append({'name':'clean_'+name,'checkpoint':str(root/'confirmatory'/name/'ema_epoch20.pt'),'group':'clean_continuation','arm':arm,'seed':seed})
    assert len(models)==25 and len({x['name'] for x in models})==25
    return models
