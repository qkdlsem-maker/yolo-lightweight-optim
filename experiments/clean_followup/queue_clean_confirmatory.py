"""A new isolated queue using the previously validated unchanged training runner."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);a=p.parse_args()
root=Path('review_20260929_clean')
assert json.loads((root/'preflight_verification.json').read_text())['passed']
for m in ['nano','large']:assert (root/'base_records'/m/'COMPLETE.json').exists()
jobs={0:[('cwd',0),('mse',1),('control',2),('cwd',2)],1:[('mse',0),('control',0),('cwd',1),('mse',2),('control',1)]}[a.device]
logroot=root/'queue_logs';logroot.mkdir(exist_ok=True)
for arm,seed in jobs:
    name=f'{arm}_seed{seed}_e20';out=root/'confirmatory'/name
    if (out/'COMPLETE.json').exists():continue
    command=[sys.executable,'-u',str(root/'confirmatory_kd.py'),'--arm',arm,'--seed',str(seed),'--device',str(a.device),'--source',str(root/'base/nano/weights/last.pt'),'--teacher',str(root/'base/large/weights/last.pt'),'--data',str(root/'clean_bdd.yaml'),'--out-root',str(root/'confirmatory')]
    if (out/'training_state.pt').exists():command+=['--resume']
    start=time.time()
    with (logroot/(name+'.log')).open('a') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record={'arm':arm,'seed':seed,'device':a.device,'exit_code':result.returncode,'seconds':time.time()-start,'complete':(out/'COMPLETE.json').exists()}
    with (logroot/f'worker{a.device}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode or not record['complete']:sys.exit(1)
print('CLEAN_QUEUE_COMPLETE',a.device,flush=True)
