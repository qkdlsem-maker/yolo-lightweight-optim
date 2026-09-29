"""Verify actual data loading and tiny objective smoke runs before full training."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
R=Path('review_20260929_clean')
manifest=json.loads((R/'clean_data_manifest.json').read_text())
for name in ['nano','large']:
    done=json.loads((R/'base_records'/(name+'_smoke')/'COMPLETE.json').read_text())
    assert done['finite_state'] and done['epochs']==1
data=check_det_dataset(str(R/'clean_bdd.yaml'));hyp=get_cfg();hyp.imgsz=640;hyp.mosaic=0
dataset=build_yolo_dataset(hyp,data['train'],64,data,mode='train',rect=False)
assert len(dataset)==69863
instances=sum(len(label['cls']) for label in dataset.labels)
# Native Ultralytics removes one exact repeated box in this verified-source
# label file; preserve the source file and explicitly account for the loader.
from collections import Counter
repeated=Counter((R/'clean_data/labels/train/75055858-7d04a650.txt').read_text().splitlines())
assert sum(v-1 for v in repeated.values())==1
assert instances==1286870,('Native training target count differs',instances)
results={};input_hashes=[]
for arm in ['control','mse','cwd']:
    out=R/'objective_smoke'/f'{arm}_seed0_e1'
    if not (out/'SMOKE_COMPLETE.json').exists():
        cmd=[sys.executable,'-u',str(R/'confirmatory_kd.py'),'--arm',arm,'--seed','0','--device','0','--epochs','1','--smoke-batches','2','--source',str(R/'base/nano_smoke/weights/last.pt'),'--teacher',str(R/'base/large_smoke/weights/last.pt'),'--data',str(R/'clean_bdd.yaml'),'--out-root',str(R/'objective_smoke')]
        with (R/'logs'/f'objective_smoke_{arm}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'SMOKE_COMPLETE.json').read_text());assert result['changed_parameter_tensors']==183 and result['max_delta']>0
    cfg=json.loads((out/'config.json').read_text());assert cfg['train_images']==69863
    results[arm]=result;input_hashes.append(result['epoch']['first_two_input_sha256'])
assert input_hashes[0]==input_hashes[1]==input_hashes[2]
record={'passed':True,'verified_train_images':len(dataset),'source_train_instances':1286871,'native_train_instances':instances,'native_loader_removed_exact_duplicate':{'image':'75055858-7d04a650','rows_removed':1,'source_file_unchanged':True},'base_smokes_finite':True,'objective_smokes':results,'paired_smoke_inputs':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'data_manifest_sha256':hashlib.sha256((R/'clean_data_manifest.json').read_bytes()).hexdigest(),'confirmatory_script_sha256':hashlib.sha256((R/'confirmatory_kd.py').read_bytes()).hexdigest()}
(R/'preflight_verification.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print('CLEAN_PHASE_PREFLIGHT_PASSED',flush=True)
