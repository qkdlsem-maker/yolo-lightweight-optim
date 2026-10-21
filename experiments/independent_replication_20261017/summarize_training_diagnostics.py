"""Summarize every frozen continuation's recorded losses and update counts.

This does not rerun training, reconstruct unavailable gradients, or compare
external scores. Epoch loss fields have different normalizations; no invented
epoch-level KD/detection objective ratio is computed.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_name(value):
    if math.isfinite(value):
        return value
    return 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')


def describe(values):
    assert len(values)==3 and all(math.isfinite(v) for v in values)
    return {'mean':statistics.mean(values),'sample_SD':statistics.stdev(values),
            'pair_values':dict(zip(['101','202','303'],values))}


def main():
    receipt_path=ROOT/'evidence_retrieval_20261020_2305.json'
    receipt=read(receipt_path)
    receipt_index={x['local_relative']:x for x in receipt['files']}
    roster=read(ROOT/'final_endpoint_roster.json')
    proof_names=sorted({x['audit_file'] for x in roster['endpoints'] if x['stage']=='continuation'})
    assert len(proof_names)==21
    rows=[]
    for name in proof_names:
        assert sha(ROOT/name)==roster['audit_sha256'][name]
        proof=read(ROOT/name);folder=ROOT/'completed_training_records'/proof['label']
        bound={}
        for path in folder.iterdir():
            relative=path.relative_to(ROOT).as_posix()
            assert sha(path)==receipt_index[relative]['sha256']
            bound[path.name]=sha(path)
        assert len(bound)==44
        config=read(folder/'config.json');delta=read(folder/'parameter_delta.json')
        records=[read(folder/f'epoch_{i:02d}.json') for i in range(1,41)]
        first=read(folder/'gradient_check.json')
        assert [x['epoch'] for x in records]==list(range(1,41))
        assert config['arm']==proof['arm'] and config['kd_weight']==proof['weight']
        assert config['batch']==64 and config['train_images']==69863
        for record in records:
            assert record['batches']==1092
            assert all(math.isfinite(record[k]) for k in ['det_loss','feature_loss','seconds','lr'])
            assert record['det_loss']>0 and record['seconds']>0
        assert math.isfinite(first['weighted_KD_to_detection_loss'])
        expected_ratio=64*proof['weight']*first['feature_loss']/first['detection_loss']
        assert math.isclose(first['weighted_KD_to_detection_loss'],expected_ratio,rel_tol=1e-6,abs_tol=1e-9)
        skipped=sum(x['optimizer_steps_skipped'] for x in records)
        applied=sum(x['batches'] for x in records)-skipped
        assert skipped==proof['optimizer_skipped_updates'] and applied==proof['optimizer_applied_updates']
        if proof['arm']=='control':
            assert first['weighted_KD_to_detection_loss']==0 and all(x['feature_loss']==0 for x in records)
        epochs=[{'epoch':r['epoch'],'lr':r['lr'],
                 'mean_batch_summed_detection_loss':r['det_loss'],
                 'mean_sum_of_feature_scale_means':r['feature_loss'],
                 'batch_attempts':r['batches'],'applied_updates':r['batches']-r['optimizer_steps_skipped'],
                 'skipped_updates':r['optimizer_steps_skipped'],
                 'recorded_median_unclipped_gradient_norm':finite_or_name(r['median_unclipped_grad_norm']),
                 'training_epoch_seconds':r['seconds']} for r in records]
        rows.append({'label':proof['label'],'pair_seed':proof['pair_seed'],'arm':proof['arm'],'weight':proof['weight'],
                     'first_batch_diagnostic':first,'epochs':epochs,
                     'undefined_or_nonfinite_gradient_median_epochs':[r['epoch'] for r in records if not math.isfinite(r['median_unclipped_grad_norm'])],
                     'attempts':applied+skipped,'applied_updates':applied,'skipped_updates':skipped,
                     'sum_recorded_training_epoch_seconds':math.fsum(r['seconds'] for r in records),
                     'trainer_elapsed_seconds_including_diagnostic_validation_and_state_saving':delta['elapsed_seconds'],
                     'source_file_sha256':bound,'training_audit_sha256':sha(ROOT/name)})
    groups={}
    for row in rows:
        groups.setdefault(row['arm']+'_w'+format(row['weight'],'g'),[]).append(row)
    group_rows=[]
    for group,members in groups.items():
        members.sort(key=lambda x:x['pair_seed'])
        assert [m['pair_seed'] for m in members]==[101,202,303]
        group_rows.append({'group':group,
            'first_batch_weighted_KD_to_detection_objective_ratio':describe([m['first_batch_diagnostic']['weighted_KD_to_detection_loss'] for m in members]),
            'fixed_epoch_loss_summaries':[{ 'epoch':epoch,
                'mean_batch_summed_detection_loss':describe([m['epochs'][epoch-1]['mean_batch_summed_detection_loss'] for m in members]),
                'mean_sum_of_feature_scale_means':describe([m['epochs'][epoch-1]['mean_sum_of_feature_scale_means'] for m in members])} for epoch in [1,20,40]]})
    prior=read(ROOT/'completed_training_audit_summary.json')
    totals={key:sum(x[key] for x in rows) for key in ['attempts','applied_updates','skipped_updates']}
    assert totals=={'attempts':prior['continuation_batch_attempts'],'applied_updates':prior['continuation_optimizer_updates_applied'],'skipped_updates':prior['continuation_optimizer_updates_skipped']}
    result={'schema':'recorded_training_diagnostics_summary_v1','status':'passed','trajectory_count':21,
            'source_sha256':sha(Path(__file__)),'retrieval_receipt_sha256':sha(receipt_path),
            'objective':'batch-summed detection loss + actual batch size * lambda * sum of per-scale feature means',
            'total_updates':totals,'trajectories':rows,'all_seven_group_summaries':group_rows,
            'limitations':[
                'Detection epoch fields average batch-summed losses; feature epoch fields average sums of per-scale means. These have different normalizations.',
                'All training batches except the last have64 images; the final batch has39. An exact epoch-weighted KD/detection ratio or mean of batch ratios cannot be reconstructed from these two epoch averages.',
                'The exact weighted objective ratio is reported only for the saved first batch. It is a loss ratio, not a gradient norm or gradient contribution ratio.',
                'First-batch gradient_check.json snapshots were byte-bound at this extraction, outside the prior full saved-state audit. No historical live gradient reconstruction is claimed.',
                'Undefined/nonfinite gradient medians are retained explicitly. Per-batch gradient norms were not archived, so their frequency cannot be reconstructed.',
                'Epoch loss means include AMP-skipped batch attempts; they are not restricted to applied updates.',
                'Recorded elapsed times are uncontrolled wall time under concurrent jobs, not isolated GPU cost or inference speed benchmarks.',
                'No outcome-dependent settings, strength selection or new optimization were introduced by this descriptive summary.'
            ]}
    target=ROOT/'completed_training_diagnostics_summary.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'passed','trajectories':21,'epochs':840,'groups':len(group_rows),
                     'updates':totals,'sha256':sha(target)}))


if __name__=='__main__':
    main()
