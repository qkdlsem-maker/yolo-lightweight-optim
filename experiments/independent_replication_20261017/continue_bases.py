"""Replace the temporary cache-recovery waiter with a bounded base scheduler.

Only scheduling changes: preserve the active large seed101 process, never repeat
the completed nano, and run the four already authorized bases on free GPUs.
After all six bases finish, invoke the unchanged frozen continuation pipeline.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

R=Path('review_20261017_independent');LOGS=R/'logs'
PLAN=[('large',202),('large',303),('nano',202),('nano',303)]

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes((json.dumps(v,indent=2)+'\n').encode());tmp.replace(p)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if alive(pid) else []
def done(model,seed):return R/'base_records'/f'{model}_seed{seed}'/'COMPLETE.json'
def integrity():
    v=json.loads((R/'PREFLIGHT_PASSED.json').read_bytes());assert v['status']=='passed'
    assert sha(R/'protocol.json')==v['protocol_sha256']
    for name,h in v['source_sha256'].items():assert sha(R/name)==h,name
    return v['protocol_sha256']

def check():
    protocol_sha=integrity();p=json.loads((R/'protocol.json').read_bytes())
    assert p['pair_seeds']==[101,202,303] and p['execution']['gpus']==[0,1]
    assert not (R/'PIPELINE_FAILURE.json').exists()
    assert not (R/'SCHEDULER_HANDOFF.json').exists(),'Handoff already attempted; inspect instead of duplicating'
    old=json.loads((R/'launch.json').read_bytes())
    assert old['mode']=='cache_recovery_waiting_for_existing_base_pair'
    assert str(R/'recover_cache.py').encode() in command(old['pid'])
    assert not alive(old['nano_pid']) and done('nano',101).exists()
    nano=json.loads(done('nano',101).read_bytes())
    assert nano['epochs']==100 and nano['finite_state'] and nano['fixed_dfl_preserved']
    assert sha(Path(nano['checkpoint']))==nano['checkpoint_sha256']
    large_cmd=command(old['large_pid'])
    assert str(R/'base_train.py').encode() in large_cmd
    for flag,value in [(b'--model',b'large'),(b'--seed',b'101'),(b'--device',b'0')]:
        assert large_cmd[large_cmd.index(flag)+1]==value
    assert not done('large',101).exists()
    for model,seed in PLAN:
        assert not (R/'base_records'/f'{model}_seed{seed}').exists()
        assert not (R/'base'/f'{model}_seed{seed}').exists()
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    gpu_map={int(line.split(',')[0]):line.split(',')[1].strip() for line in gpu_rows}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    assert not any(line.split(',')[-1].strip()==gpu_map[1] for line in apps if line.strip()),'GPU1 has another compute job'
    return old,{'utc':now(),'protocol_sha256':protocol_sha,'scheduler_sha256':sha(Path(__file__)),
        'completed_nano_checkpoint_sha256':nano['checkpoint_sha256'],
        'old_coordinator_pid':old['pid'],'preserved_large_pid':old['large_pid'],
        'remaining_bases':PLAN,'scope':'Scheduling only; no source/data/seed/epochs/objective changes. Long jobs are dispatched before short jobs to reduce idle time.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    guard=(R/'base_scheduler.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old,record=check()
    if args.check_only:print(json.dumps(record,indent=2));return
    dump(R/'SCHEDULER_HANDOFF.json',record)
    # Signal only the verified temporary coordinator, never a training process.
    os.kill(old['pid'],signal.SIGTERM)
    for _ in range(100):
        if not alive(old['pid']):break
        time.sleep(.1)
    assert not alive(old['pid']) and alive(old['large_pid'])
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={0:{'model':'large','seed':101,'pid':old['large_pid'],'process':None},1:None}
    pending=list(PLAN);journal=[]
    try:
        while pending or any(active.values()):
            for device,job in list(active.items()):
                if job is None:continue
                process=job['process']
                running=alive(job['pid']) if process is None else process.poll() is None
                if running:continue
                assert done(job['model'],job['seed']).exists(),f'Base stopped without completion: {job}'
                if process is not None:assert process.returncode==0
                journal.append({'event':'completed','device':device,'model':job['model'],'seed':job['seed'],'pid':job['pid'],'utc':now()})
                active[device]=None
            for device in [0,1]:
                if active[device] is not None or not pending:continue
                integrity();model,seed=pending.pop(0);name=f'base_{model}_seed{seed}'
                assert not (R/'base_records'/f'{model}_seed{seed}').exists()
                cmd=[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed),'--device',str(device)]
                with (LOGS/(name+'.log')).open('ab') as log:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[device]={'model':model,'seed':seed,'pid':process.pid,'process':process}
                dump(LOGS/(name+'_active.json'),{'pid':process.pid,'command':cmd,'started_utc':now(),'coordinator':'continue_bases.py'})
                journal.append({'event':'started','device':device,'model':model,'seed':seed,'pid':process.pid,'utc':now()})
                print('STARTED',name,process.pid,'DEVICE',device,flush=True)
            snapshot={str(d):{k:v for k,v in job.items() if k!='process'} for d,job in active.items() if job}
            dump(R/'launch.json',{'pid':os.getpid(),'mode':'bounded_base_scheduler_after_cache_recovery','active_bases':snapshot,'pending':pending,'utc':now()})
            dump(R/'base_scheduler_journal.json',journal)
            if pending or active[0] or active[1]:time.sleep(15)
        for seed in [101,202,303]:
            for model in ['nano','large']:assert done(model,seed).exists()
        dump(R/'CACHE_RECOVERY_COMPLETE.json',{'status':'complete','first_pair_complete':True,'all_six_bases_complete':True,'utc':now()})
        dump(R/'BASE_SCHEDULER_COMPLETE.json',{'status':'complete','jobs':6,'utc':now()})
    except Exception as error:
        dump(R/'PIPELINE_FAILURE.json',{'status':'failed','error':str(error),'source':'continue_bases.py','utc':now()})
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    # The frozen queue skips all six completed bases and runs its original 21 arms.
    with (LOGS/'pipeline.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(R/'run_pipeline.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        dump(R/'launch.json',{'pid':process.pid,'mode':'normal_coordinator_after_all_six_bases','utc':now()})
        code=process.wait()
    if code:raise SystemExit(code)

if __name__=='__main__':main()
