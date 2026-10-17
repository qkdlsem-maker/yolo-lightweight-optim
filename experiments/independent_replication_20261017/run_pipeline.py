"""Durable bounded experiment queue; completion and file locks prevent duplicate runs."""
import concurrent.futures,fcntl,hashlib,json,subprocess,sys,time,threading
from pathlib import Path
R=Path('review_20261017_independent');P=json.loads((R/'protocol.json').read_text());logs=R/'logs';logs.mkdir(exist_ok=True)
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def run(name,cmd):
    with (logs/(name+'.log')).open('a') as f:
        process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        dump(logs/(name+'_active.json'),{'pid':process.pid,'command':cmd,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print('STARTED',name,process.pid,flush=True);code=process.wait()
    print('FINISHED',name,code,flush=True)
    if code:raise RuntimeError(f'{name} exited {code}')
def integrity():
    v=json.loads((R/'PREFLIGHT_PASSED.json').read_text());assert v['status']=='passed'
    assert hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()==v['protocol_sha256']
    for n,h in v['source_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
def stage(name,jobs):
    pending=iter(jobs);guard=threading.Lock();stop=threading.Event()
    def worker(device):
        while not stop.is_set():
            with guard:job=next(pending,None)
            if job is None:return
            label,command,done,resumable=job
            if done.exists():continue
            integrity();cmd=command+['--device',str(device)]
            if resumable.exists():cmd+=['--resume']
            try:run(label,cmd);assert done.exists(),label
            except Exception:stop.set();raise
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(worker,d) for d in P['execution']['gpus']]
        for f in futures:f.result()
    dump(R/(name+'_COMPLETE.json'),{'status':'complete','jobs':len(jobs)})
try:
    if not (R/'PREFLIGHT_PASSED.json').exists():run('preflight',[sys.executable,'-u',str(R/'preflight.py')])
    integrity()
    jobs=[]
    for seed in P['pair_seeds']:
        for model in ['large','nano']:
            name=f'{model}_seed{seed}'
            jobs.append(('base_'+name,[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed)],
              R/'base_records'/name/'COMPLETE.json',R/'base'/name/'weights/last.pt'))
    stage('BASE_TRAINING',jobs)
    jobs=[]
    for seed in P['pair_seeds']:
        for arm,weight in [('control',0.)]+[(a,w) for a in ['mse','cwd'] for w in P['continuation']['positive_weights']]:
            label=f'pair{seed}_{arm}_w{weight:g}';outroot=R/'continuations'/label;data_seed=seed+1000
            out=outroot/f'{arm}_seed{data_seed}_e40'
            cmd=[sys.executable,'-u',str(R/'continuation.py'),'--arm',arm,'--seed',str(data_seed),'--epochs','40',
              '--batch','64','--workers','4','--lr','.0001','--kd-weight',str(weight),'--temperature','4',
              '--source',str(R/'base'/f'nano_seed{seed}'/'weights/last.pt'),
              '--teacher',str(R/'base'/f'large_seed{seed}'/'weights/last.pt'),
              '--data',P['data']['yaml'],'--out-root',str(outroot)]
            jobs.append((label,cmd,out/'COMPLETE.json',out/'training_state.pt'))
    stage('CONTINUATION_TRAINING',jobs)
    dump(R/'TRAINING_COMPLETE.json',{'status':'complete','base_runs':6,'continuation_runs':21,
      'remaining':'independent numerical and input audits, fixed external and common-metric evaluations, contribution review and manuscript/GitHub update'})
except Exception as e:
    dump(R/'PIPELINE_FAILURE.json',{'status':'failed','error':str(e),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    raise
