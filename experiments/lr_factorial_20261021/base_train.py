"""Independently seeded native base training; old inputs are read only."""
import argparse,csv,hashlib,json,time,fcntl,random
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import seed_worker
from step_accounting import OptimizerStepAccounting

R=Path('review_20261021_lr_factorial')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
def dump(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)

class EpochSeededLoader(torch.utils.data.DataLoader):
    """Fresh workers and an explicit generator for each epoch, including resume."""
    def set_epoch(self,epoch):self.generator.manual_seed(self.run_seed+100003*epoch)
    def reset(self):pass  # Native close-mosaic reset; workers are recreated on iteration.

class SeededDetectionTrainer(OptimizerStepAccounting,DetectionTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=-1,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        assert rank==-1,'This protocol uses one process per GPU, not DDP'
        dataset=self.build_dataset(dataset_path,mode,batch_size)
        loader=EpochSeededLoader(dataset,batch_size=min(batch_size,len(dataset)),shuffle=True,
          num_workers=self.args.workers,pin_memory=True,collate_fn=dataset.collate_fn,
          worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(self.args.seed),persistent_workers=False)
        loader.run_seed=self.args.seed
        return loader

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['nano','large'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    protocol=json.loads((R/'protocol.json').read_text());b=protocol['base']
    assert a.seed in protocol['pair_seeds']
    name=f'{a.model}_seed{a.seed}'+('_smoke' if a.smoke else '')
    record=R/'base_records'/name;record.mkdir(parents=True,exist_ok=True)
    lock=(record/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (record/'COMPLETE.json').exists():print('ALREADY_COMPLETE',name,flush=True);return
    source=Path(protocol['previous_root'])/'pretrained'/('yolov8n.pt' if a.model=='nano' else 'yolov8l.pt')
    assert sha(source)==b[a.model+'_source_sha256']
    assert sha(protocol['data']['yaml'])==protocol['data']['yaml_sha256']
    data=str(R/'smoke.yaml') if a.smoke else protocol['data']['yaml']
    config={'name':name,'model':a.model,'seed':a.seed,'device':a.device,'source':str(source),'source_sha256':sha(source),
      'data':data,'data_sha256':sha(data),'script_sha256':sha(__file__),'protocol_sha256':sha(R/'protocol.json'),
      'epochs':1 if a.smoke else b['epochs'],'fraction':1.0,'batch':b['batch'][a.model],'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if (record/'config.json').exists():
        prior=json.loads((record/'config.json').read_text())
        for k in config.keys()-{'created_utc'}:assert prior[k]==config[k],k
        assert a.resume,'Incomplete run requires inspected explicit resume'
    else:dump(record/'config.json',config)
    run=R/'base'/name;checkpoint=run/'weights/last.pt';torch.set_num_threads(4)
    traces={};holder={};handles=[]
    def start(trainer):
        if a.resume:
            state=torch.load(record/'resume_state.pt',map_location=trainer.device,weights_only=False)
            assert state['epoch']==trainer.start_epoch
            trainer.model.load_state_dict(state['model']);trainer.ema.ema.load_state_dict(state['ema'])
            trainer.ema.updates=state['ema_updates'];trainer.optimizer.load_state_dict(state['optimizer'])
            trainer.scaler.load_state_dict(state['scaler']);trainer.scheduler.load_state_dict(state['scheduler'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu(),trainer.device)
        trainer.install_step_accounting(state['actual_step_counts'] if a.resume else None)
        initial={k:{'shape':list(v.shape),'sha256':tensor_sha(v)} for k,v in trainer.model.named_parameters()}
        if not (record/'initial_parameters.json').exists():dump(record/'initial_parameters.json',initial)
        dump(record/'effective_config.json',vars(trainer.args))
        holder['trainer']=trainer
        def capture(module,args):
            if not module.training or not args or not isinstance(args[0],dict):return
            epoch=trainer.epoch+1;values=traces.setdefault(epoch,[])
            if len(values)>=2:return
            h=hashlib.sha256()
            for k in ['img','cls','bboxes','batch_idx']:
                t=args[0][k].detach().cpu().contiguous();h.update(k.encode());h.update(t.numpy().tobytes())
            values.append(h.hexdigest())
        handles.append(trainer.model.register_forward_pre_hook(capture))
    def progress(trainer):
        loss=trainer.tloss.detach().cpu();assert torch.isfinite(loss).all()
        epoch=trainer.epoch+1
        # Final evaluation after checkpoint stripping can repeat this callback.
        if epoch in traces:
            assert len(traces[epoch])==2,(name,epoch,traces[epoch])
            dump(record/f'epoch_{epoch:03d}.json',{'epoch':epoch,'loss_items':loss.tolist(),'first_two_input_sha256':traces[epoch],
              'actual_step_counts':dict(trainer.actual_step_counts),'native_EMA_updates':trainer.ema.updates,
              'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            state={'epoch':epoch,'model':trainer.model.state_dict(),'ema':trainer.ema.ema.state_dict(),
              'ema_updates':trainer.ema.updates,'actual_step_counts':dict(trainer.actual_step_counts),'optimizer':trainer.optimizer.state_dict(),'scaler':trainer.scaler.state_dict(),
              'scheduler':trainer.scheduler.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(trainer.device)}
            tmp=record/'resume_state.pt.tmp';torch.save(state,tmp);tmp.replace(record/'resume_state.pt')
        dump(record/'progress.json',{'epoch':epoch,'epochs':config['epochs'],'loss_items':loss.tolist(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if a.resume:
        assert checkpoint.exists();model=YOLO(str(checkpoint))
    else:
        assert not run.exists();model=YOLO(str(source))
    model.add_callback('on_train_start',start);model.add_callback('on_fit_epoch_end',progress)
    model.add_callback('on_train_epoch_start',lambda t:t.train_loader.set_epoch(t.epoch))
    if a.resume:model.train(trainer=SeededDetectionTrainer,resume=True,device=a.device)
    else:
        model.train(trainer=SeededDetectionTrainer,data=data,epochs=config['epochs'],batch=config['batch'],nbs=b['nbs'],device=a.device,imgsz=b['imgsz'],
          seed=a.seed,deterministic=b['deterministic'],optimizer=b['optimizer'],lr0=b['lr0'],lrf=b['lrf'],cos_lr=b['cos_lr'],
          momentum=b['momentum'],weight_decay=b['weight_decay'],warmup_epochs=b['warmup_epochs'],patience=b['patience'],
          close_mosaic=0 if a.smoke else b['close_mosaic'],amp=b['amp'],workers=2 if a.smoke else b['workers'],
          pretrained=True,project=str(R/'base'),name=name,exist_ok=False,save=True,save_period=25,cache=False,plots=False,verbose=False,val=True)
    for h in handles:h.remove()
    rows=list(csv.DictReader((run/'results.csv').open()));assert len(rows)==config['epochs']
    saved=YOLO(str(checkpoint)).model;assert saved.model[-1].nc==10
    assert all(torch.isfinite(v).all() for v in saved.state_dict().values())
    initial=json.loads((record/'initial_parameters.json').read_text())
    changed=[k for k,v in saved.named_parameters() if tensor_sha(v)!=initial[k]['sha256']]
    fixed=[k for k in initial if 'dfl.conv.weight' in k]
    assert changed and fixed and not set(fixed)&set(changed)
    assert len(list(record.glob('epoch_*.json')))==config['epochs']
    dump(record/'COMPLETE.json',{'status':'complete','seed':a.seed,'model':a.model,'epochs':config['epochs'],
      'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(source),
      'initial_parameters_sha256':sha(record/'initial_parameters.json'),'finite_state':True,'changed_parameter_tensors':len(changed),
      'fixed_dfl_preserved':True,'actual_step_counts':dict(holder['trainer'].actual_step_counts),'native_EMA_updates':holder['trainer'].ema.updates,'last_row':rows[-1],'endpoint':'final epoch EMA, never best validation epoch'})
    print('BASE_COMPLETE',name,flush=True)
if __name__=='__main__':main()
