"""Read-only audit of saved training states and AMP update accounting.

An AMP overflow may skip an optimizer step without corrupting a model. Preserve
the original training records and distinguish a nonfinite diagnostic reduction
from nonfinite saved weights. This script does not change the experiment.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch

def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for child in value.values():yield from tensors(child)
    elif isinstance(value,(list,tuple)):
        for child in value:yield from tensors(child)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='review_20260929')
    p.add_argument('--include-active',action='store_true')
    a=p.parse_args();root=Path(a.root);torch.set_num_threads(1);records={}
    runner_digest=hashlib.sha256((root/'confirmatory_kd.py').read_bytes()).hexdigest()
    for folder in sorted((root/'confirmatory').iterdir()):
        complete=(folder/'COMPLETE.json').exists()
        if not complete and not a.include_active:continue
        checkpoint=folder/'training_state.pt'
        if not checkpoint.exists():continue
        # The runner writes this file by atomic replacement, so an open load
        # observes one complete epoch even if the next checkpoint is being saved.
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        epoch=int(state['epoch'])
        epochs=[json.loads((folder/f'epoch_{e:02d}.json').read_text()) for e in range(1,epoch+1)]
        cfg=json.loads((folder/'config.json').read_text())
        assert cfg['script_sha256']==runner_digest,(folder.name,'runner changed')
        nonfinite={};counts={}
        for key in ['student','adapters','ema','optimizer']:
            ts=list(tensors(state[key]));counts[key]=len(ts)
            nonfinite[key]=sum(not bool(torch.isfinite(t).all()) for t in ts)
        losses_finite=all(math.isfinite(e[k]) for e in epochs for k in ['det_loss','feature_loss'])
        bad_norm_epochs=[e['epoch'] for e in epochs if not math.isfinite(e['median_unclipped_grad_norm'])]
        unaccounted=[e['epoch'] for e in epochs if not math.isfinite(e['median_unclipped_grad_norm']) and e['optimizer_steps_skipped']==0]
        attempted=sum(e['batches'] for e in epochs)
        skipped=sum(e['optimizer_steps_skipped'] for e in epochs)
        applied=attempted-skipped
        assert max(nonfinite.values())==0,(folder.name,nonfinite)
        assert losses_finite and not unaccounted,(folder.name,'nonfinite loss or unaccounted diagnostic')
        assert state['ema_updates']==applied,(folder.name,'EMA/update accounting mismatch')
        if complete:assert epoch==20
        records[folder.name]={
            'complete':complete,'snapshot_epoch':epoch,'tensor_counts':counts,
            'nonfinite_tensor_counts':nonfinite,'all_recorded_objective_means_finite':losses_finite,
            'attempted_minibatches':attempted,'skipped_optimizer_steps':skipped,
            'applied_optimizer_steps':applied,'EMA_updates':state['ema_updates'],
            'AMP_scale_at_snapshot':state['scaler']['scale'],
            'nonfinite_median_gradient_norm_epochs':bad_norm_epochs,
            'every_nonfinite_gradient_norm_epoch_has_recorded_AMP_skip':not unaccounted,
            'executed_script_hash_matches_config':True}
        del state
    result={'snapshot_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'status':'passed','includes_active_runs':a.include_active,'runs':records,
            'interpretation':'AMP overflows recorded by GradScaler can skip updates; saved student, adapter, EMA and optimizer tensors remain finite. Original np.median diagnostics can be NaN if a skipped-step gradient norm was nonfinite. Report actual applied steps, not an assertion of identical optimizer-update counts. This snapshot does not establish future checkpoint validity.'}
    (root/'numerical_state_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
