"""Bounded subprocess execution shared by the new phase only."""
from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time,uuid,shutil
R=Path('review_20261021_lr_factorial')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(p)+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def integrity():
    p=read(R/'protocol.json');m=read(R/'deployed_sources.json')
    for x in m['files']:assert sha(R/x['name'])==x['sha256'],x['name']
    assert p['root']==str(R)
    return p
def budget_used():
    now=time.time();seconds=0.;start=now
    for file in (R/'attempts').glob('*.json'):
        x=read(file);start=min(start,x['started_unix'])
        if x['GPU_job']:seconds+=max(0,x.get('ended_unix',now)-x['started_unix'])
    return seconds/3600,(now-start)/86400
def budget_check():
    p=read(R/'protocol.json')['execution'];h,d=budget_used()
    assert h<p['hard_allocated_GPU_hours'],f'GPU budget reached: {h:.3f} hours'
    assert d<p['hard_wall_days'],f'Wall budget reached: {d:.3f} days'
    assert shutil.disk_usage('.').free/2**30>=p['min_free_disk_GiB'],'Free-disk floor reached'
def gpu_idle(device):
    gpu=subprocess.check_output(['nvidia-smi',f'--id={device}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    return not any(row.split(',')[-1].strip()==gpu for row in rows if row.strip())
def run_job(label,cmd,device=None):
    integrity();budget_check()
    if device is not None:assert gpu_idle(device),f'GPU{device} is busy; no competing process is terminated'
    attempt=label+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6]
    (R/'attempts').mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
    path=R/'attempts'/(attempt+'.json');env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env['YOLO_CONFIG_DIR']=str((R/'config').absolute())
    record={'label':label,'command':cmd,'GPU_job':device is not None,'device':device,'started_utc':utc(),'started_unix':time.time(),'status':'starting','log':str(R/'logs'/(attempt+'.log'))}
    with Path(record['log']).open('xb') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        record.update(pid=proc.pid,status='running');dump(path,record)
        print('STARTED',label,proc.pid,'device',device,flush=True)
        try:
            while proc.poll() is None:
                time.sleep(10);budget_check()
        except BaseException as error:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            record['stop_reason']=repr(error)
            raise
        finally:
            record.update(ended_unix=time.time(),ended_utc=utc(),returncode=proc.poll(),status='completed' if proc.returncode==0 else 'failed')
            dump(path,record)
    assert proc.returncode==0,f'{label} exited{proc.returncode}; preserve {path}'
    print('FINISHED',label,flush=True)
    return record
