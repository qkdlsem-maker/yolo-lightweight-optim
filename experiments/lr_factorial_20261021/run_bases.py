"""Only the six new bases. Continuation requires a separate completed preflight."""
import concurrent.futures,fcntl,sys,threading,traceback
from pathlib import Path
from job_runtime import R,read,sha,dump,utc,run_job,integrity
lock=(R/'base_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=integrity();check=read(R/'BASE_PREFLIGHT_PASSED.json')
assert check['status']=='passed' and check['protocol_sha256']==sha(R/'protocol.json')
assert check['deployed_sources_sha256']==sha(R/'deployed_sources.json')
assert read(R/'PROTOCOL_PUBLICATION.json')['status']=='passed'
pending=iter([(model,seed) for model in ['large','nano'] for seed in p['pair_seeds']]);guard=threading.Lock();stop=threading.Event()
def worker(device):
    while not stop.is_set():
        with guard:job=next(pending,None)
        if job is None:return
        model,seed=job;folder=R/'base_records'/f'{model}_seed{seed}'
        if (folder/'COMPLETE.json').exists():continue
        assert not (folder/'config.json').exists(),'Inspect incomplete state and authorize a technical resume separately'
        try:
            run_job(f'base_{model}_{seed}',[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed),'--device',str(device)],device)
            c=read(folder/'COMPLETE.json');assert c['epochs']==100 and c['finite_state'] and c['actual_step_counts']['applied']>0
        except BaseException:stop.set();raise
try:
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(worker,d) for d in [0,1]]
        for f in futures:f.result()
    dump(R/'BASE_TRAINING_COMPLETE.json',{'status':'complete','utc':utc(),'new_base_runs':6,'remaining':'saved-state audit and gated factorial continuation; this is not whole-task completion'})
except BaseException as e:
    dump(R/'PIPELINE_FAILURE.json',{'utc':utc(),'stage':'new_bases','error':repr(e),'traceback':traceback.format_exc()});raise
