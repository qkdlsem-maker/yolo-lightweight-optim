"""Hold detector predictions fixed while adding/removing zero-area legacy targets."""
import argparse,json
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import ap_per_class
p=argparse.ArgumentParser();p.add_argument('--root',default='review_20260929');a=p.parse_args();root=Path(a.root)
zero=json.loads((root/'zero_area_val_targets.json').read_text());results={}
for folder in sorted((root/'stratified').iterdir()):
 if not (folder/'metrics.json').exists():continue
 with np.load(folder/'image_statistics.npz',allow_pickle=False) as z:
  assert len(z['images'])==10000
  extra=np.array([c for name in z['images'] for c in zero.get(str(name),[])],dtype=z['target_cls'].dtype)
  assert len(extra)==52 and len(z['target_cls'])==185526
  values={k:z[k] for k in ['tp','conf','pred_cls','target_cls']}
  clean=ap_per_class(**values,plot=False)[5]
  values['target_cls']=np.concatenate([values['target_cls'],extra])
  original=ap_per_class(**values,plot=False)[5]
  metrics=json.loads((folder/'metrics.json').read_text())
  assert abs(clean.mean()-metrics['overall']['AP50_95_present_classes'])<1e-12
  results[folder.name]={'original_target_count':185578,'reconstructed_target_count':185526,'original_AP50_same_predictions':float(original[:,0].mean()),'original_AP50_95_same_predictions':float(original.mean()),'reconstructed_AP50':float(clean[:,0].mean()),'reconstructed_AP50_95':float(clean.mean()),'delta_AP50_pp':float(100*(clean[:,0].mean()-original[:,0].mean())),'delta_AP50_95_pp':float(100*(clean.mean()-original.mean()))}
out={'method':'Same predictions and IoU true-positive flags; 52 zero-area targets cannot have positive IoU and only change target denominators. A full original-label baseline evaluation provides an independent check.','models':results}
(root/'annotation_sensitivity.json').write_text(json.dumps(out,indent=2));print('ANNOTATION_SENSITIVITY',len(results),flush=True)
