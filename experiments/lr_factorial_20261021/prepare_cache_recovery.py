"""Preserve the failed pre-optimizer attempt and serially warm native label caches."""
from pathlib import Path
import fcntl,json,shutil,sys
import torch
from job_runtime import R,read,sha,dump,utc,integrity
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
torch.set_num_threads(4)
p=integrity();dest=R/'technical_failures/002_native_label_cache_race'
assert not dest.exists();dest.mkdir(parents=True)
attempts=[x for x in (R/'attempts').glob('base_large_505_*.json')]
assert len(attempts)==1
a=read(attempts[0]);assert a['status']=='failed' and a['returncode']==1
assert not Path(f"/proc/{a['pid']}/cmdline").exists()
log=Path(a['log']);text=log.read_text()
assert 'FileNotFoundError' in text and 'train.cache.npy' in text
record=R/'base_records/large_seed505';run=R/'base/large_seed505'
assert sorted(x.name for x in record.iterdir())==['config.json','run.lock']
assert {str(x.relative_to(run)) for x in run.rglob('*') if x.is_file()}=={'args.yaml'}
lock=(record/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan={'utc':utc(),'cause':'Two native dataset constructors rebuilt the shared full-training label cache after subset smoke checks; their temporary .cache.npy rename raced. Seed505 exited during dataset setup, before on_train_start, input batches or optimizer updates.',
 'action':'Preserve the entire failed attempt. Warm the complete train/val caches serially, then run seed505 with unchanged source/seed/conditions on idle GPU1. Seed404 continues under its original coordinator. No training state exists for505 to resume.',
 'ownership':'Recovery worker initially owns only GPU1 and unstarted/failed505 plus unstarted remaining bases. GPU0 is eligible only after the old coordinator exits and seed404 completes. Old coordinator stop flag was set by failed505, so it will dispatch no more jobs.',
 'training_source_sha256':sha(R/'base_train.py'),'protocol_sha256':sha(R/'protocol.json'),'attempt_sha256':sha(attempts[0]),
 'budget':'All failed and recovery subprocess wall time remains charged. No completed training repeated.'}
dump(dest/'recovery_plan.json',plan)
for src in [attempts[0],log,R/'run_bases.py',R/'deployed_sources.json']:
 shutil.copy2(src,dest/src.name);assert sha(src)==sha(dest/src.name)
for src,name in [(record,'original_base_record'),(run,'original_run')]:
 manifest={str(x.relative_to(src)):sha(x) for x in src.rglob('*') if x.is_file()}
 src.rename(dest/name)
 assert manifest=={str(x.relative_to(dest/name)):sha(x) for x in (dest/name).rglob('*') if x.is_file()}
 dump(dest/(name+'_manifest.json'),manifest)
d=check_det_dataset(p['data']['yaml']);hyp=get_cfg();hyp.imgsz=640
counts={}
for split,mode,batch in [('train','train',16),('val','val',32)]:
 ds=build_yolo_dataset(hyp,d[split],batch,d,mode=mode,rect=mode=='val',stride=32)
 counts[split]={'images':len(ds),'boxes':sum(len(x['cls']) for x in ds.labels)}
assert counts=={'train':{'images':69863,'boxes':1286870},'val':{'images':10000,'boxes':185526}},counts
dump(dest/'CACHE_READY.json',{'utc':utc(),'status':'passed','counts':counts,'original_labels_unchanged':'Native derived cache only; labels separately hash-verified in data_verification.json.'})
print('CACHE_RECOVERY_READY',flush=True)
