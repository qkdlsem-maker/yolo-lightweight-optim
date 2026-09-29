"""Read-only compact status, safe to call while training is active."""
import json,time
from pathlib import Path
R=Path('review_20260929_clean')
report={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'base':{},'continuations':{},'markers':{}}
for model in ['nano','large']:
    root=R/'base_records'/model
    state={'started':(root/'config.json').exists(),'complete':(root/'COMPLETE.json').exists()}
    if (root/'progress.json').exists():state['progress']=json.loads((root/'progress.json').read_text())
    log=R/'logs'/('base_'+model+'.log')
    if log.exists():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1000));state['log_tail']=f.read().decode('utf-8',errors='replace')
    report['base'][model]=state
for p in sorted((R/'confirmatory').glob('*_e20')):
    epochs=sorted(p.glob('epoch_*.json'))
    report['continuations'][p.name]={'epoch_records':len(epochs),'complete':(p/'COMPLETE.json').exists()}
for name in ['preflight_verification.json','PIPELINE_FAILURE.json','TRAINING_PIPELINE_COMPLETE.json','external_kitti/DOWNLOAD_COMPLETE.json','external_kitti/evaluator_verification.json','POSTPROCESS_COMPLETE.json']:
    report['markers'][name]=(R/name).exists()
partial=R/'external_kitti/data_object_image_2.zip.partial'
if partial.exists():report['kitti_download_bytes']=partial.stat().st_size
print(json.dumps(report,indent=2,ensure_ascii=False))
