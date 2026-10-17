"""Recover a native cache-creation race without interrupting the valid nano run."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path('review_20261017_independent');logs=R/'logs'
lock=(R/'cache_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def checked_pid(name):
    pid=json.loads((logs/(name+'_active.json')).read_text())['pid']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes() if alive(pid) else b''
    assert b'base_train.py' in cmd and b'--seed\x00101' in cmd,(name,pid)
    return pid
assert not (R/'CACHE_RECOVERY_COMPLETE.json').exists()
oldpid=json.loads((R/'launch.json').read_text())['pid']
assert b'run_pipeline.py' in Path(f'/proc/{oldpid}/cmdline').read_bytes()
nano=checked_pid('base_nano_seed101')
large_old=json.loads((logs/'base_large_seed101_active.json').read_text())['pid']
assert not alive(large_old),'Large process still active; do not duplicate'
assert not (R/'base_records/large_seed101/initial_parameters.json').exists()
assert not (R/'base/large_seed101/results.csv').exists()
# Stop only the failed coordinator. The separately running nano child continues.
os.kill(oldpid,signal.SIGTERM)
for _ in range(100):
    if not alive(oldpid):break
    time.sleep(.1)
assert not alive(oldpid) and alive(nano)
pipeline_lock=(R/'pipeline.lock').open('a');fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
archive=R/'technical_failures/003_cache_race';archive.mkdir(parents=True,exist_ok=False)
for n in ['base_records/large_seed101','base/large_seed101']:
    p=R/n;assert p.resolve().is_relative_to(R.resolve())
    p.rename(archive/(p.parent.name+'_'+p.name))
for n in ['launch.json','PIPELINE_FAILURE.json']:
    p=R/n
    if p.exists():p.rename(archive/n)
reason={'reason':'Concurrent initial full-data cache reconstruction after smoke subsets collided on train.cache.npy. No large training step occurred. Valid nano training remains running. Reuse its completed full-data caches, restart only the never-trained large job, then attach a fresh unchanged coordinator after both finish.',
  'nano_pid_preserved':nano,'failed_large_pid':large_old,'old_coordinator_pid':oldpid,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
dump(archive/'reason.json',reason)
data=Path('review_20260929_clean/clean_data/labels')
assert (data/'train.cache').exists() and (data/'val.cache').exists()
name='base_large_seed101'
with (logs/(name+'.log')).open('a') as f:
    cmd=[sys.executable,'-u',str(R/'base_train.py'),'--model','large','--seed','101','--device','0']
    large=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    dump(logs/(name+'_active.json'),{'pid':large.pid,'command':cmd,'technical_retry':'never-trained cache-race job'})
    dump(R/'launch.json',{'pid':os.getpid(),'mode':'cache_recovery_waiting_for_existing_base_pair','nano_pid':nano,'large_pid':large.pid,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('RECOVERY_ATTACHED_NANO',nano,'RESTARTED_LARGE',large.pid,flush=True)
    while True:
        done={m:(R/'base_records'/f'{m}_seed101'/'COMPLETE.json').exists() for m in ['nano','large']}
        if all(done.values()):break
        if (not done['nano'] and not alive(nano)) or (not done['large'] and large.poll() is not None):
            dump(R/'PIPELINE_FAILURE.json',{'error':'Base process failed during cache recovery','done':done});raise RuntimeError('Inspect base logs')
        time.sleep(20)
dump(R/'CACHE_RECOVERY_COMPLETE.json',{'status':'complete','first_pair_complete':True})
fcntl.flock(pipeline_lock,fcntl.LOCK_UN);pipeline_lock.close()
with (logs/'pipeline.log').open('a') as f:
    p=subprocess.Popen([sys.executable,'-u',str(R/'run_pipeline.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    dump(R/'launch.json',{'pid':p.pid,'mode':'normal_coordinator_after_verified_cache_recovery'})
    code=p.wait()
if code:raise SystemExit(code)
