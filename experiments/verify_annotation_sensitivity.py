"""Check the fixed-prediction annotation calculation against direct validation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='review_20260929');a=p.parse_args();root=Path(a.root)
s=json.loads((root/'annotation_sensitivity.json').read_text())['models']['yolov8n_baseline']
d=json.loads((root/'original_label_check/yolov8n_baseline/metrics.json').read_text())
assert d['annotation_set']=='original-local' and d['batch']==8
assert sum(d['overall']['instances_by_class'])==185578
errors={k:abs(s['original_'+k+'_same_predictions']-d['overall'][k+'_present_classes']) for k in ['AP50','AP50_95']}
assert max(errors.values())<1e-12,errors
result={'passed':True,'direct_original_label_evaluation':'original_label_check/yolov8n_baseline/metrics.json','absolute_AP_differences':errors,'tolerance':1e-12,'scope':'Independent baseline check only; all models use the same target-denominator calculation.'}
(root/'annotation_sensitivity_check.json').write_text(json.dumps(result,indent=2));print(result)
