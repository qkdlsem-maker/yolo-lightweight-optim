"""Read-only public arithmetic check from all 48 hash-bound numeric proofs.

No import of the aggregation code and no model inference or source mutation.
"""
import json, hashlib, math
from pathlib import Path
from datetime import datetime, timezone

R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
A = read(R/'final_all_pair_analysis.json')
C = read(R/'final_evaluation/EVALUATION_COMPLETE.json')
V = read(R/'final_evaluation_audits/AUDITS_COMPLETE.json')
assert sha(R/'final_evaluation/EVALUATION_COMPLETE.json') == A['evaluation_complete_sha256']
assert sha(R/'final_evaluation_audits/AUDITS_COMPLETE.json') == A['numeric_audits_complete_sha256']
assert sha(R/'aggregate_final_evaluation_v2.py') == A['aggregation_source_sha256']
assert len(A['endpoint_rows']) == len(C['endpoint_complete_sha256']) == len(V['proof_sha256']) == 48
idx = {r['endpoint_id']: r for r in A['endpoint_rows']}
assert set(idx) == set(C['endpoint_complete_sha256']) == set(V['proof_sha256'])
checks = 0
def eq(x,y):
    global checks
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y) < 1e-12, (x,y)
    checks += 1
for name,row in idx.items():
    p=R/'final_evaluation_audits'/f'{name}.json'
    proof=read(p)
    assert sha(p)==V['proof_sha256'][name]==row['numeric_audit_sha256']
    assert proof['status']=='passed' and proof['audit_source_sha256']==V['audit_source_sha256']
    assert proof['plan_sha256']==A['plan_sha256']==C['plan_sha256']==V['plan_sha256']
    assert sha(R/'evaluation_endpoint_markers'/f'{name}.json')==C['endpoint_complete_sha256'][name]==proof['endpoint_complete_sha256']
    for metric,value in row['metrics_fraction'].items():
        if metric.startswith('bdd_native_'):
            key={'AP50_95':'metrics/mAP50-95(B)','AP50':'metrics/mAP50(B)','precision':'metrics/precision(B)','recall':'metrics/recall(B)'}[metric[11:]]
            expected=proof['native_bdd']['results'][key]
        elif metric.startswith('kitti_native_'):
            tail=metric[13:]
            if tail=='mean_moderate_AP_R40': expected=proof['native_kitti']['mean_moderate_AP_R40']
            else:
                cat,diff=tail.removesuffix('_AP_R40').split('_')
                expected=proof['native_kitti']['class_AP_R40'][cat][diff]
        else:
            variants={**proof['bdd_common']['scores'],**proof['kitti_common']['scores']}
            variant=next(v for v in variants if metric.startswith(v+'_'))
            tail=metric[len(variant)+1:].removesuffix('AP50_95').rstrip('_')
            expected=variants[variant]['class_AP50_95'][tail] if tail else variants[variant]['mean_AP50_95']
        eq(value,expected)
def check_summary(s,values):
    assert s['n_pairs']==3
    mean=sum(values)/3
    sd=math.sqrt(sum((v-mean)**2 for v in values)/2)
    eq(s['mean_pp'],mean); eq(s['sample_SD_pp'],sd)
    for seed,v in zip([101,202,303],values): eq(s['pair_values_pp'][str(seed)],v)
for group in A['all_group_summaries']:
    rows=[idx[n] for n in group['endpoint_ids']]
    assert [r['pair_seed'] for r in rows]==[101,202,303]
    for metric,s in group['metrics'].items(): check_summary(s,[100*r['metrics_fraction'][metric] for r in rows])
signs=[]
for contrast in A['all_distillation_vs_control_contrasts']:
    for metric,s in contrast['metrics'].items():
        values=[]
        for pair in contrast['pair_differences']:
            t,c=idx[pair['treatment']],idx[pair['control']]
            assert t['pair_seed']==c['pair_seed']==pair['pair_seed']
            assert t['epoch']==c['epoch']==contrast['epoch'] and c['arm']=='control'
            assert t['arm']==contrast['arm'] and t['weight']==contrast['weight']
            difference=t['metrics_fraction'][metric]-c['metrics_fraction'][metric]
            eq(pair['delta_fraction'][metric],difference); values.append(100*difference)
        check_summary(s,values)
    if contrast['epoch']==40:
        s=contrast['metrics']['bdd_native_AP50_95']
        signs.append({'arm':contrast['arm'],'weight':contrast['weight'],**s})
result={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'verification_source_sha256':sha(Path(__file__)),
        'analysis_sha256':sha(R/'final_all_pair_analysis.json'),'endpoint_count':48,'metric_values':960,
        'group_summaries':16,'contrasts':12,'numeric_comparisons':checks,'absolute_tolerance':1e-12,
        'method':'Direct local proof extraction; elementary sums and sample variance. No aggregation-code import, training or inference.',
        'primary_pair_effects_pp':signs}
print(json.dumps(result,indent=2))
