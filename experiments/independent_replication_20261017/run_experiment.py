"""Reproduction entry point: prime full-data caches serially after smoke tests."""
import fcntl,json,subprocess,sys
from pathlib import Path
R=Path('review_20261017_independent')
lock=(R/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (R/'TRAINING_COMPLETE.json').exists():raise SystemExit('Already complete; do not repeat experiments')
if not (R/'PREFLIGHT_PASSED.json').exists():
    subprocess.run([sys.executable,'-u',str(R/'preflight.py')],check=True)
# Smoke image lists share the reconstructed label directory in this fixed source
# version. Warm both complete caches before two native processes read them.
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
p=json.loads((R/'protocol.json').read_text());d=check_det_dataset(p['data']['yaml'])
h=get_cfg();h.imgsz=640
for mode,split,count in [('train','train',69863),('val','val',10000)]:
    dataset=build_yolo_dataset(h,d[split],64 if mode=='train' else 32,d,mode=mode,rect=mode=='val')
    assert len(dataset)==count
subprocess.run([sys.executable,'-u',str(R/'run_pipeline.py')],check=True)
