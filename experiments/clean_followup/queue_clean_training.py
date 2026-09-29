"""Durable two-GPU base training, then nine paired continuations and audits.

Do not rerun completed experiments. A failure is recorded and stops dependent
work. Rerunning this coordinator can resume saved checkpoints under identical
settings, after the failure has been inspected.
"""
import fcntl,json,subprocess,sys,time
from pathlib import Path
R=Path('review_20260929_clean');logs=R/'logs';logs.mkdir(exist_ok=True)
lock=(R/'training_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((R/'preflight_verification.json').read_text())['passed']

def group(stage,commands):
    active=[];failed=[]
    for name,cmd in commands:
        handle=(logs/(name+'.log')).open('a');process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
        active.append((name,process,handle));print('STARTED',stage,name,process.pid,flush=True)
    for name,process,handle in active:
        code=process.wait();handle.close()
        if code:failed.append({'job':name,'exit_code':code})
        print('EXIT',stage,name,code,flush=True)
    if failed:
        record={'status':'failed','stage':stage,'failures':failed,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        (R/'PIPELINE_FAILURE.json').write_text(json.dumps(record,indent=2));raise SystemExit(1)

commands=[]
for model,device in [('nano',0),('large',1)]:
    rec=R/'base_records'/model
    if (rec/'COMPLETE.json').exists():continue
    cmd=[sys.executable,'-u',str(R/'clean_base_train.py'),'--model',model,'--device',str(device)]
    if (R/'base'/model/'weights/last.pt').exists():cmd.append('--resume')
    commands.append(('base_'+model,cmd))
group('fresh_base_models',commands)
for model in ['nano','large']:assert (R/'base_records'/model/'COMPLETE.json').exists()
group('nine_continuations',[(f'continuation_worker{device}',[sys.executable,'-u',str(R/'queue_clean_confirmatory.py'),'--device',str(device)]) for device in [0,1]])
group('verification',[
    ('summary',[sys.executable,'-u',str(R/'summarize_clean_confirmatory.py'),'--root',str(R)]),
    ('numerical_audit',[sys.executable,'-u',str(R/'audit_confirmatory_numerics.py'),'--root',str(R)])])
(R/'TRAINING_PIPELINE_COMPLETE.json').write_text(json.dumps({'status':'complete','base_models':2,'continuation_models':9,'followup':'External KITTI, illumination evaluation and manuscript update still required'},indent=2))
print('ALL_CLEAN_TRAINING_AND_PRIMARY_AUDITS_COMPLETE',flush=True)
