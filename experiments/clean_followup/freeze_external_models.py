"""Freeze all 25 prespecified artifacts before any real KITTI inference."""
import json,time
from pathlib import Path
from clean_followup_utils import sha,dump,model_roster
R=Path('review_20260929_clean');external=R/'external_kitti'
assert (R/'TRAINING_PIPELINE_COMPLETE.json').exists()
assert json.loads((external/'input_audit.json').read_text())['status']=='passed'
assert json.loads((external/'evaluator_verification.json').read_text())['passed']
assert json.loads((external/'inference_io_verification.json').read_text())['passed']
models=model_roster()
for row in models:
    row['checkpoint_sha256']=sha(row['checkpoint'])
    if 'prior_hash_record' in row:
        prior=json.loads(Path(row['prior_hash_record']).read_text());assert prior['checkpoint_sha256']==row['checkpoint_sha256'],row['name']
    else:
        p=Path(row['checkpoint']).parent
        assert (p/'COMPLETE.json').exists() and (p/'ema_epoch20_metrics.json').exists()
        a=json.loads((p/'COMPLETE.json').read_text());b=json.loads((p/'ema_epoch20_metrics.json').read_text());assert a['state_sha256']==b['state_sha256']
        assert b['checkpoint_sha256']==row['checkpoint_sha256']
manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(R/'protocol.md'),'input_audit_sha256':sha(external/'input_audit.json'),'inference_script_sha256':sha(R/'external_kitti_inference.py'),'evaluator_executable_sha256':sha(external/'local_evaluator/evaluate_2d_local'),'models':models,'inference':{'imgsz':640,'rect':False,'batch':8,'half':False,'conf':.001,'iou':.7,'max_det':300,'classes_before_nms':'all ten BDD classes','retained_after_nms':{'0':'Pedestrian','2':'Car'}}}
dest=external/'frozen_models.json'
if dest.exists():
    prior=json.loads(dest.read_text());assert all(prior[k]==v for k,v in manifest.items() if k!='created_utc')
else:
    assert not list((external/'results').glob('*/metrics.json')),'Real external scores exist before artifact freeze'
    dump(dest,manifest)
print('ALL_25_EXTERNAL_ARTIFACTS_FROZEN',flush=True)
