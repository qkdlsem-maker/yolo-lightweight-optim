"""Durable held-out and illumination evaluation after frozen training endpoints."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from clean_followup_utils import dump,model_roster
R=Path('review_20260929_clean');logs=R/'evaluation_logs';logs.mkdir(exist_ok=True)

def run(name,command):
    with (logs/(name+'.log')).open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('VERIFIED_STAGE_COMPLETE',name,flush=True)

def worker(stage,device):
    roster=model_roster()
    if stage=='illumination':roster=[r for r in roster if r['group'] in ['clean_base','clean_continuation']]
    for row in roster[device::2]:
        name=row['name']
        if stage=='external':
            if (R/'external_kitti/results'/name/'metrics.json').exists():continue
            command=[sys.executable,'-u',str(R/'external_kitti_inference.py'),'--name',name,'--device',str(device)]
        else:
            if (R/'stratified'/name/'metrics.json').exists():continue
            command=[sys.executable,'-u',str(R/'stratified_validation.py'),'--checkpoint',row['checkpoint'],'--name',name,'--device',str(device),'--batch','8','--data',str(R/'clean_bdd.yaml'),'--attributes','review_20260929/legacy_val_attributes.json','--out-root',str(R/'stratified'),'--annotation-set','Reconstructed legacy 2018 source labels; 185526 positive-area validation boxes']
        run(stage+'_'+name,command)

def wait_for(file,stage):
    deadline=time.time()+10*24*3600
    while not file.exists():
        if time.time()>deadline:raise RuntimeError('Dependency wait timeout: '+stage)
        if (R/'PIPELINE_FAILURE.json').exists():raise RuntimeError('Training pipeline reported a technical failure; inspect before recovery')
        time.sleep(60)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['external','illumination']);p.add_argument('--device',type=int,choices=[0,1]);a=p.parse_args()
    if a.worker:assert a.device is not None;worker(a.worker,a.device);return
    lock=(R/'postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        wait_for(R/'external_kitti/DOWNLOAD_COMPLETE.json','KITTI input download')
        if not (R/'external_kitti/input_audit.json').exists():run('input_audit',[sys.executable,'-u',str(R/'audit_external_kitti_inputs.py')])
        wait_for(R/'TRAINING_PIPELINE_COMPLETE.json','two fresh base models and nine continuations with audits')
        run('freeze_models',[sys.executable,'-u',str(R/'freeze_external_models.py')])
        for stage in ['external','illumination']:
            jobs=[]
            for device in [0,1]:
                handle=(logs/f'{stage}_worker{device}.log').open('a')
                proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',stage,'--device',str(device)],stdout=handle,stderr=subprocess.STDOUT);jobs.append((proc,handle))
            failures=[]
            for process,handle in jobs:
                code=process.wait();handle.close()
                if code:failures.append(code)
            assert not failures,(stage,failures)
        models=model_roster()
        assert all((R/'external_kitti/results'/row['name']/'metrics.json').exists() for row in models)
        clean=[row for row in models if row['group'] in ['clean_base','clean_continuation']]
        assert len(clean)==11 and all((R/'stratified'/row['name']/'metrics.json').exists() for row in clean)
        dump(R/'POSTPROCESS_COMPLETE.json',{'status':'complete','external_models':25,'clean_label_illumination_models':11,'followup':'Scientific analysis, cross-checks, manuscript/GitHub update and all-page visual QA remain required'})
        print('ALL_EXTERNAL_AND_ILLUMINATION_EVALUATIONS_COMPLETE',flush=True)
    except Exception as e:
        dump(R/'POSTPROCESS_FAILURE.json',{'status':'failed','type':type(e).__name__,'error':str(e),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});raise

if __name__=='__main__':main()
