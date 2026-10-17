"""Check independent native initialization and paired objective inputs before full runs."""
import concurrent.futures,hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path('review_20261017_independent');logs=R/'logs';logs.mkdir(exist_ok=True)
def run(name,cmd):
    with (logs/(name+'.log')).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('PREFLIGHT_JOB_COMPLETE',name,flush=True)
run('data_verification',[sys.executable,'-u',str(R/'setup_data.py')])
run('cwd_unit',[sys.executable,'-u',str(R/'test_cwd.py')])
def smoke_base(model,device):
    for seed in [101,202]:
        name=f'{model}_seed{seed}_smoke'
        if not (R/'base_records'/name/'COMPLETE.json').exists():
            run(name,[sys.executable,'-u',str(R/'base_train.py'),'--model',model,'--seed',str(seed),'--device',str(device),'--smoke'])
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    futures=[pool.submit(smoke_base,'nano',0),pool.submit(smoke_base,'large',1)]
    for f in futures:f.result()
checks={}
for model in ['nano','large']:
    records=[R/'base_records'/f'{model}_seed{s}_smoke' for s in [101,202]]
    initial=[json.loads((p/'initial_parameters.json').read_text()) for p in records]
    assert initial[0]!=initial[1],f'{model}: seed did not change the initialized task head'
    inputs=[json.loads((p/'epoch_001.json').read_text())['first_two_input_sha256'] for p in records]
    assert inputs[0]!=inputs[1],f'{model}: base input seeds coincide'
    for p in records:
        v=json.loads((p/'COMPLETE.json').read_text());assert v['finite_state'] and v['fixed_dfl_preserved'] and v['changed_parameter_tensors']>0
    checks[model]={'distinct_initial_parameters':True,'distinct_actual_inputs':True,'finite_updates':True}
results={};all_inputs=[]
for arm,weight in [('control',0.),('mse',.03),('mse',.3),('mse',3.),('cwd',.03),('cwd',.3),('cwd',3.)]:
    label=f'{arm}_w{weight:g}';outroot=R/'objective_smoke'/label;out=outroot/f'{arm}_seed1101_e1'
    if not (out/'SMOKE_COMPLETE.json').exists():
        run(label,[sys.executable,'-u',str(R/'continuation.py'),'--arm',arm,'--seed','1101','--device','0','--epochs','1','--smoke-batches','2',
          '--kd-weight',str(weight),'--workers','2','--source',str(R/'base/nano_seed101_smoke/weights/last.pt'),
          '--teacher',str(R/'base/large_seed101_smoke/weights/last.pt'),'--data',str(R/'smoke.yaml'),'--out-root',str(outroot)])
    v=json.loads((out/'SMOKE_COMPLETE.json').read_text());assert v['changed_parameter_tensors']==183 and v['max_delta']>0
    assert v['epoch']['batches']-v['epoch']['optimizer_steps_skipped']>0
    all_inputs.append(v['epoch']['first_two_input_sha256']);results[label]=v
assert all(x==all_inputs[0] for x in all_inputs)
for name in ['base_train.py','continuation.py','setup_data.py','preflight.py','run_pipeline.py','test_cwd.py','protocol.json']:
    assert (R/name).exists()
record={'status':'passed','base_seed_checks':checks,'objective_smokes':results,'paired_actual_smoke_inputs':True,
 'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),
 'source_sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['base_train.py','continuation.py','setup_data.py','preflight.py','run_pipeline.py','test_cwd.py']},
 'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(R/'PREFLIGHT_PASSED.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print('ALL_PREFLIGHT_CHECKS_PASSED',flush=True)
