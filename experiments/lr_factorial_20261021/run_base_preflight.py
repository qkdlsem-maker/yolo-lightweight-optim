import concurrent.futures,fcntl,json,sys,traceback
from pathlib import Path
import torch
from job_runtime import R,read,sha,dump,utc,run_job,integrity
lock=(R/'preflight.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'BASE_PREFLIGHT_PASSED.json').exists(),'Already completed; do not repeat'
try:
    integrity()
    assert read('review_20261017_independent/TRAINING_COMPLETE.json')['status']=='complete'
    assert not Path('review_20261017_independent/PIPELINE_FAILURE.json').exists()
    reuse=read(R/'historical_reuse_manifest.json')
    for row in reuse['endpoints']:assert sha(row['checkpoint']['path'])==row['checkpoint']['sha256']
    run_job('verify_data',[sys.executable,'-u',str(R/'setup_data.py')])
    run_job('native_step_count_parity',[sys.executable,'-u',str(R/'test_step_accounting.py')],0)
    def smokes(model,device):
        for seed in [404,505]:
            done=R/'base_records'/f'{model}_seed{seed}_smoke'/'COMPLETE.json'
            if not done.exists():run_job(f'smoke_{model}_{seed}',[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed),'--device',str(device),'--smoke'],device)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(smokes,'nano',0),pool.submit(smokes,'large',1)]
        for f in futures:f.result()
    checks={}
    for model in ['nano','large']:
        records=[R/'base_records'/f'{model}_seed{s}_smoke' for s in [404,505]]
        assert read(records[0]/'initial_parameters.json')!=read(records[1]/'initial_parameters.json')
        assert read(records[0]/'epoch_001.json')['first_two_input_sha256']!=read(records[1]/'epoch_001.json')['first_two_input_sha256']
        counts=[]
        for folder in records:
            done=read(folder/'COMPLETE.json');c=done['actual_step_counts']
            assert done['finite_state'] and done['fixed_dfl_preserved'] and done['changed_parameter_tensors']>0
            assert c['attempted']==c['applied']+c['skipped'] and c['applied']>0,c
            state=torch.load(folder/'resume_state.pt',map_location='cpu',weights_only=False)
            assert state['actual_step_counts']==c
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert all(torch.isfinite(v).all() for v in state['ema'].values())
            counts.append(c)
        checks[model]={'seed_initialization_differs':True,'actual_input_hashes_differ':True,'finite_raw_and_EMA_states':True,'actual_step_counts':counts}
    dump(R/'BASE_PREFLIGHT_PASSED.json',{'status':'passed','utc':utc(),'protocol_sha256':sha(R/'protocol.json'),'deployed_sources_sha256':sha(R/'deployed_sources.json'),'base_checks':checks,'step_accounting_validation_sha256':sha(R/'step_accounting_validation.json'),'historical24endpoints_unchanged':True,'continuation_ready':False})
    print('BASE_PREFLIGHT_PASSED',flush=True)
except BaseException as e:
    dump(R/'BASE_PREFLIGHT_FAILURE.json',{'utc':utc(),'error':repr(e),'traceback':traceback.format_exc()})
    raise
