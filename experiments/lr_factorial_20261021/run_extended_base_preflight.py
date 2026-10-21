"""One fixed extended AMP fixture; no full-training/source condition is changed."""
import concurrent.futures,fcntl,json,sys,traceback,shutil,hashlib
from pathlib import Path
import torch,yaml
from job_runtime import R,read,sha,dump,utc,run_job,integrity
from base_train import tensor_sha
lock=(R/'preflight.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'BASE_PREFLIGHT_PASSED.json').exists()
F=R/'technical_failures/001_short_native_AMP_fixture';F.mkdir(parents=True,exist_ok=True)
for name in ['BASE_PREFLIGHT_FAILURE.json','preflight.log','run_base_preflight.py','deployed_sources.json']:
    dst=F/name
    if dst.exists():assert dst.read_bytes()==(R/name).read_bytes()
    else:shutil.copyfile(R/name,dst)
E=R/'extended_smoke';E.mkdir(exist_ok=True)
try:
    integrity();assert read(R/'step_accounting_validation.json')['status']=='passed'
    # Fixed before this single recovery; no trial-length search.
    addon={'reason':'128-image native smoke had only2 nano optimizer attempts, both skipped by initial native AMP scaling. The real step hook exposed this; checkpoint quantization differences cannot establish raw updates.',
     'fixture_train_images':2048,'fixture_val_images':32,'epochs':1,'architectures':['nano','large'],'seeds':[404,505],
     'stop':'One fixed fixture per architecture/seed. If no applied updates or finite raw changes, stop for investigation; do not extend again based on performance.',
     'scientific_conditions_changed':False,'old_smoke_outputs_retained':True,'utc':utc()}
    if not (F/'recovery_plan.json').exists():dump(F/'recovery_plan.json',addon)
    if not (E/'protocol.json').exists():shutil.copyfile(R/'protocol.json',E/'protocol.json')
    assert sha(E/'protocol.json')==sha(R/'protocol.json')
    cfg=yaml.safe_load(Path(read(R/'protocol.json')['data']['yaml']).read_text())
    old=Path(read(R/'protocol.json')['previous_root'])
    for split,n in [('train',2048),('val',32)]:
        imgs=sorted((old/'clean_data/images'/split).glob('*.jpg'),key=lambda p:hashlib.sha256(('preflight20261017:'+p.name).encode()).hexdigest())[:n]
        assert len(imgs)==n
        raw='\n'.join(str(p.absolute()) for p in imgs)+'\n';dst=E/f'smoke_{split}.txt'
        if dst.exists():assert dst.read_text()==raw
        else:dst.write_text(raw)
    cfg={'path':str(Path.cwd()),'train':str((E/'smoke_train.txt').absolute()),'val':str((E/'smoke_val.txt').absolute()),'names':cfg['names']}
    raw=yaml.safe_dump(cfg,sort_keys=False)
    if (E/'smoke.yaml').exists():assert (E/'smoke.yaml').read_text()==raw
    else:(E/'smoke.yaml').write_text(raw)
    def smokes(model,device):
        for seed in [404,505]:
            done=E/'base_records'/f'{model}_seed{seed}_smoke'/'COMPLETE.json'
            if not done.exists():
                cmd=[sys.executable,'-u',str(R/'run_extended_smoke.py'),'--model',model,'--seed',str(seed),'--device',str(device),'--smoke']
                run_job(f'extended_smoke_{model}_{seed}',cmd,device)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(smokes,'nano',0),pool.submit(smokes,'large',1)]
        for f in futures:f.result()
    checks={}
    for model in ['nano','large']:
        records=[E/'base_records'/f'{model}_seed{s}_smoke' for s in [404,505]]
        assert read(records[0]/'initial_parameters.json')!=read(records[1]/'initial_parameters.json')
        assert read(records[0]/'epoch_001.json')['first_two_input_sha256']!=read(records[1]/'epoch_001.json')['first_two_input_sha256']
        counts=[]
        for folder in records:
            done=read(folder/'COMPLETE.json');c=done['actual_step_counts'];initial=read(folder/'initial_parameters.json')
            assert c['attempted']==c['applied']+c['skipped'] and c['applied']>0,c
            state=torch.load(folder/'resume_state.pt',map_location='cpu',weights_only=False)
            assert state['actual_step_counts']==c
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert all(torch.isfinite(v).all() for v in state['ema'].values())
            changed=[k for k,v in initial.items() if tensor_sha(state['model'][k])!=v['sha256']]
            assert changed and not any('dfl.conv.weight' in k for k in changed)
            counts.append({'counts':c,'changed_raw_parameter_tensors':len(changed)})
        checks[model]={'seed_initialization_differs':True,'actual_inputs_differ':True,'finite_raw_and_EMA':True,'runs':counts}
    dump(R/'BASE_PREFLIGHT_PASSED.json',{'status':'passed','utc':utc(),'protocol_sha256':sha(R/'protocol.json'),
      'deployed_sources_sha256':sha(R/'deployed_sources.json'),'recovery_sources_sha256':sha(R/'recovery_sources.json'),
      'base_checks':checks,'step_accounting_validation_sha256':sha(R/'step_accounting_validation.json'),
      'historical24endpoints_unchanged':True,'data_verification_sha256':sha(R/'data_verification.json'),'continuation_ready':False})
    dump(F/'RESOLVED.json',{'status':'resolved','utc':utc(),'base_preflight_sha256':sha(R/'BASE_PREFLIGHT_PASSED.json'),'full_training_not_started_at_failure':True,'scientific_sources_or_conditions_changed':False})
    print('BASE_PREFLIGHT_PASSED',flush=True)
except BaseException as e:
    dump(R/'EXTENDED_PREFLIGHT_FAILURE.json',{'utc':utc(),'error':repr(e),'traceback':traceback.format_exc()});raise
