"""All-pair descriptive summaries; refuses an incomplete fixed evaluation/audit set.

No best strength/epoch selection, tests of significance, historical pooling,
new inference, or writes to source evidence. AP values are fractions in raw
rows; contrasts and displayed summaries are percentage points.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    assert len(values) == 3 and all(math.isfinite(x) for x in values)
    return {'n_pairs': 3, 'mean_pp': 100*statistics.mean(values),
            'sample_SD_pp': 100*statistics.stdev(values),
            'pair_values_pp': dict(zip(['101', '202', '303'], [100*x for x in values]))}


def extract(proof):
    native = proof['native_bdd']['results']
    metrics = {'bdd_native_AP50_95': native['metrics/mAP50-95(B)'],
               'bdd_native_AP50': native['metrics/mAP50(B)'],
               'bdd_native_precision': native['metrics/precision(B)'],
               'bdd_native_recall': native['metrics/recall(B)']}
    for part in ['bdd_common', 'kitti_common']:
        for variant, row in proof[part]['scores'].items():
            metrics[variant+'_AP50_95'] = row['mean_AP50_95']
            for category, value in row['class_AP50_95'].items():
                metrics[variant+'_'+category+'_AP50_95'] = value
    native_kitti = proof['native_kitti']
    metrics['kitti_native_mean_moderate_AP_R40'] = native_kitti['mean_moderate_AP_R40']
    for category, values in native_kitti['class_AP_R40'].items():
        for difficulty, value in values.items():
            metrics['kitti_native_'+category+'_'+difficulty+'_AP_R40'] = value
    assert len(metrics) == 20
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in metrics.values())
    return metrics


def aggregate(plan_path, expected_sha, audit_root):
    assert sha(plan_path) == expected_sha
    plan = read(plan_path)
    root = Path(plan['output_root'])
    completed = read(root/'EVALUATION_COMPLETE.json')
    audits = read(audit_root/'AUDITS_COMPLETE.json')
    assert completed['plan_sha256'] == audits['plan_sha256'] == expected_sha
    assert audits['status'] == 'passed'
    assert sha(plan['roster']['path']) == plan['roster']['sha256']
    roster = read(plan['roster']['path'])['endpoints']
    assert len(roster) == len(plan['jobs']) == len(audits['proof_sha256']) == 48
    ids = {x['endpoint_id'] for x in roster}
    assert ids == set(audits['proof_sha256']) == set(completed['endpoint_complete_sha256'])
    assert ids == {x['endpoint_id'] for x in plan['jobs']}
    jobs = {x['endpoint_id']: x for x in plan['jobs']}
    rows = []
    for endpoint in roster:
        key = endpoint['endpoint_id']; folder = Path(jobs[key]['output'])
        proof_path = audit_root/(key+'.json')
        assert sha(proof_path) == audits['proof_sha256'][key]
        proof = read(proof_path)
        assert proof['status'] == 'passed' and proof['endpoint_id'] == key
        assert proof['plan_sha256'] == expected_sha
        assert proof['audit_source_sha256'] == audits['audit_source_sha256']
        assert proof['endpoint_complete_sha256'] == completed['endpoint_complete_sha256'][key] == sha(folder/'ENDPOINT_COMPLETE.json')
        for step, proof_key in [('native_bdd', 'native_bdd'), ('kitti_native', 'native_kitti')]:
            assert sha(folder/step/'score.json') == proof[proof_key]['score_sha256']
        for part in ['bdd_common', 'kitti_common']:
            for variant, score in proof[part]['scores'].items():
                assert sha(folder/part/(variant+'.json')) == score['score_sha256']
        rows.append({**endpoint, 'metrics_fraction': extract(proof), 'numeric_audit_sha256': sha(proof_path)})

    names = list(rows[0]['metrics_fraction'])
    assert all(list(x['metrics_fraction']) == names for x in rows)
    groups = {}
    for row in rows:
        if row['stage'] == 'base':
            group = 'base_'+row['architecture']+'_e100'
        else:
            group = row['arm']+'_w'+format(row['weight'], 'g')+'_e'+str(row['epoch'])
        groups.setdefault(group, []).append(row)
    assert len(groups) == 16
    group_results = []
    for group, members in groups.items():
        members.sort(key=lambda x: x['pair_seed'])
        assert [x['pair_seed'] for x in members] == [101, 202, 303]
        group_results.append({'group': group, 'role': members[0]['role'],
                              'endpoint_ids': [x['endpoint_id'] for x in members],
                              'metrics': {name: summary([x['metrics_fraction'][name] for x in members]) for name in names}})

    index = {x['endpoint_id']: x for x in rows}
    contrasts = []
    for epoch in [40, 20]:
        for arm in ['mse', 'cwd']:
            for weight in [.03, .3, 3.]:
                paired = []
                for seed in [101, 202, 303]:
                    treatment = index[f'pair{seed}_{arm}_w{weight:g}_e{epoch}']
                    control = index[f'pair{seed}_control_w0_e{epoch}']
                    assert treatment['pair_seed'] == control['pair_seed'] == seed
                    assert treatment['epoch'] == control['epoch'] == epoch
                    paired.append({'pair_seed':seed, 'treatment':treatment['endpoint_id'], 'control':control['endpoint_id'],
                                   'delta_fraction':{n:treatment['metrics_fraction'][n]-control['metrics_fraction'][n] for n in names}})
                contrasts.append({'arm':arm, 'weight':weight, 'epoch':epoch,
                                  'role':'primary' if epoch == 40 else 'secondary_same_40epoch_schedule',
                                  'pair_differences':paired,
                                  'metrics':{n:summary([x['delta_fraction'][n] for x in paired]) for n in names}})
    assert len(contrasts) == 12 and sum(len(x['pair_differences']) for x in contrasts) == 36

    return {'schema':'fixed_all_pair_analysis_v1', 'plan_sha256':expected_sha,
            'roster_sha256':plan['roster']['sha256'], 'aggregation_source_sha256':sha(__file__),
            'evaluation_complete_sha256':sha(root/'EVALUATION_COMPLETE.json'),
            'numeric_audits_complete_sha256':sha(audit_root/'AUDITS_COMPLETE.json'),
            'endpoint_count':48, 'primary_metric':'bdd_native_AP50_95',
            'raw_metric_scale':'fraction_0_to_1', 'summary_scale':'percentage_points',
            'pair_seeds':[101,202,303], 'endpoint_rows':rows,
            'all_group_summaries':group_results, 'all_distillation_vs_control_contrasts':contrasts,
            'interpretation_limits':[
                'Three training pairs; sample SD is not a precise population uncertainty estimate, confidence interval, equivalence test or significance claim.',
                'The same three pairs contribute all strengths and both epochs; neither 21 trajectories nor 48 endpoints are independent replicates.',
                'Epoch20 is nested in the frozen forty-epoch schedule. Epoch40 is primary. No endpoint, strength or favorable pair is selected.',
                'BDD validation is reused and based on a reconstructed legacy-label mirror; KITTI is the previously examined public labeled training partition.',
                'Common metrics harmonize classes, IoU grid and integration but not remaining annotation, ignore or image-population differences.',
                'Common KITTI ignore/no-ignore scores and native KITTI difficulty-specific scores are distinct definitions and must be named separately.',
                'Baseline teacher quality or cross-dataset score differences alone do not establish causal explanations for distillation behavior.',
                'No new inference, checkpoint optimization, external-score tuning, historical pooling or significance/equivalence tests were performed by this analysis.'
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.plan, args.plan_sha256, args.audit_root)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status':'complete','endpoints':48,'paired_contrasts':12,'output_sha256':sha(args.output)}))


if __name__ == '__main__':
    main()
