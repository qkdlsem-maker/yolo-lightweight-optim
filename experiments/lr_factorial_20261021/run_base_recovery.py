"""Own only the five unfinished bases; never duplicate the original seed404 worker."""
import concurrent.futures,fcntl,sys,threading,time,traceback
from pathlib import Path
from job_runtime import R,read,sha,dump,utc,run_job,integrity,budget_check
lock=(R/'base_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=integrity();dest=R/'technical_failures/002_native_label_cache_race'
assert read(dest/'CACHE_READY.json')['status']=='passed'
assert read(R/'BASE_PREFLIGHT_PASSED.json')['protocol_sha256']==sha(R/'protocol.json')
original=read(R/'launch.json');oldpid=original['pid']
pending=iter([('large',505),('large',606),('nano',404),('nano',505),('nano',606)])
guard=threading.Lock();stop=threading.Event()
def gate():
 assert not (R/'BUDGET_STOP.json').exists(),'Budget stop is latched'
 budget_check()
def alive(pid):
 f=Path(f'/proc/{pid}/cmdline')
 return f.exists() and bool(f.read_bytes())
def worker(device):
 if device==0:
  while alive(oldpid):
   gate()
   if stop.is_set():return
   time.sleep(30)
  assert (R/'base_records/large_seed404/COMPLETE.json').exists(),'Original seed404 did not finish; inspect its saved state'
  legacy_lock=(R/'base_pipeline.lock').open('a');fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not stop.is_set():
  gate()
  with guard:job=next(pending,None)
  if job is None:return
  model,seed=job;folder=R/'base_records'/f'{model}_seed{seed}'
  assert not (folder/'config.json').exists(),'Do not repeat an initialized job; inspect a technical resume separately'
  try:
   run_job(f'recovery_base_{model}_{seed}',[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed),'--device',str(device)],device)
   c=read(folder/'COMPLETE.json');assert c['epochs']==100 and c['finite_state'] and c['actual_step_counts']['applied']>0
  except BaseException:stop.set();raise
try:
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  futures=[pool.submit(worker,d) for d in [1,0]]
  for f in futures:f.result()
 for model in ['nano','large']:
  for seed in p['pair_seeds']:
   c=read(R/'base_records'/f'{model}_seed{seed}'/'COMPLETE.json')
   assert c['epochs']==100 and c['finite_state'] and c['actual_step_counts']['applied']>0
 dump(dest/'RESOLVED.json',{'status':'resolved','utc':utc(),'all_six_new_bases_complete':True,'original_failure_retained':True})
 dump(R/'BASE_TRAINING_COMPLETE.json',{'status':'complete','utc':utc(),'new_base_runs':6,'remaining':'Independent saved-state audit and separately gated continuations; not whole-task completion.'})
except BaseException as error:
 dump(R/'BASE_RECOVERY_FAILURE.json',{'utc':utc(),'error':repr(error),'traceback':traceback.format_exc()});raise
