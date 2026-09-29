"""Independent-data, paired continuation with MSE/CWD and a no-KD control.

Run from the original project root. This supersedes neither historical outputs
nor the September 22 fixed-data runs. CWD implements Shu et al., ICCV 2021,
Eq. (3), as channel-averaged spatial KL with temperature scaling.
"""
import argparse, csv, hashlib, json, math, os, random, sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path.cwd()/'review_20260929/config'))
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset, seed_worker
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import init_seeds, ModelEMA

LAYERS = [15, 18, 21]

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def dump(path, obj):
    tmp = Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)

def features(model, image):
    values = {}; handles = []
    for i in LAYERS:
        handles.append(model.model[i].register_forward_hook(lambda m, a, o, i=i: values.__setitem__(i, o)))
    try:
        model(image)
    finally:
        for h in handles: h.remove()
    return [values[i] for i in LAYERS]

def cwd_loss(student, teacher, temperature=4.):
    """KL(teacher spatial distribution || student), mean over N and C."""
    if student.shape != teacher.shape:
        raise ValueError('CWD requires aligned N,C,H,W shapes')
    s = F.log_softmax(student.float().flatten(2)/temperature, dim=-1)
    t = F.log_softmax(teacher.detach().float().flatten(2)/temperature, dim=-1)
    return (t.exp()*(t-s)).sum(-1).mean()*(temperature**2)

def make_loader(dataset, batch, workers, seed, epoch):
    # Reconstruct each epoch so restart behavior and all worker seeds are explicit.
    g = torch.Generator().manual_seed(seed + 100003*epoch)
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=True, collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker, generator=g, persistent_workers=False)

def input_digest(batch):
    h=hashlib.sha256()
    for k in ['img','cls','bboxes','batch_idx']:
        h.update(k.encode());h.update(batch[k].contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_save(obj, path):
    tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);tmp.replace(path)

def evaluate(source, model_state, out, device, data, name):
    fresh=YOLO(source);fresh.model.load_state_dict(model_state)
    fresh.save(str(out/(name+'.pt')))
    m=fresh.val(data=data,imgsz=640,batch=32,device=device,workers=4,
        half=False,plots=False,verbose=False,project=str(out),name=name+'_validation')
    result={'map50':float(m.box.map50),'map50_95':float(m.box.map),
        'precision':float(m.box.mp),'recall':float(m.box.mr),'classes':m.names,
        'ap50':m.box.ap50.tolist(),'ap50_95':m.box.ap.tolist(),
        'checkpoint_sha256':file_hash(out/(name+'.pt')),'state_sha256':state_hash(model_state)}
    dump(out/(name+'_metrics.json'),result)
    del fresh;torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['control','mse','cwd'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--kd-weight',type=float,default=.3);p.add_argument('--temperature',type=float,default=4.)
    p.add_argument('--source',default='outputs/yolov8n_baseline/weights/best.pt')
    p.add_argument('--teacher',default='outputs/yolov8l_baseline/weights/best.pt')
    p.add_argument('--data',default='configs/bdd100k.yaml')
    p.add_argument('--out-root',default='review_20260929/confirmatory')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-batches',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(args.device)
    init_seeds(args.seed,deterministic=True)
    name=f'{args.arm}_seed{args.seed}_e{args.epochs}';out=Path(args.out_root)/name
    if (out/'COMPLETE.json').exists():
        print('ALREADY_COMPLETE',name,flush=True);return
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    sy=YOLO(args.source);student=sy.model.to(args.device)
    teacher=YOLO(args.teacher).model.to(args.device).eval()
    for x in teacher.parameters():x.requires_grad_(False)
    for n,x in student.named_parameters():x.requires_grad_('dfl.conv.weight' not in n)
    hyp=get_cfg(DEFAULT_CFG);hyp.imgsz=640;hyp.seed=args.seed;hyp.deterministic=True
    # Conservative post-convergence continuation, fixed before results.
    hyp.mosaic=0.;hyp.mixup=0.;hyp.copy_paste=0.
    student.args=hyp;teacher.args=hyp;student.eval()
    before_buffers={k:v.detach().cpu().clone() for k,v in student.named_buffers()}
    with torch.no_grad():
        x=torch.zeros(1,3,640,640,device=args.device)
        sc=[v.shape[1] for v in features(student,x)];tc=[v.shape[1] for v in features(teacher,x)]
    assert all(torch.equal(v.detach().cpu(),before_buffers[k]) for k,v in student.named_buffers())
    adapters=torch.nn.ModuleList([torch.nn.Conv2d(a,b,1) for a,b in zip(sc,tc)]).to(args.device)
    ema=ModelEMA(student)
    fs={};ft={};handles=[]
    for model,store in [(student,fs),(teacher,ft)]:
        for i in LAYERS:
            handles.append(model.model[i].register_forward_hook(lambda m,a,o,i=i,store=store:store.__setitem__(i,o)))
    parameters=[p for p in student.parameters() if p.requires_grad]+list(adapters.parameters())
    optimizer=torch.optim.SGD(parameters,lr=args.lr,momentum=.937,weight_decay=.0005)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    init_seeds(args.seed,deterministic=True)
    data=check_det_dataset(args.data);dataset=build_yolo_dataset(hyp,data['train'],args.batch,data,mode='train',rect=False)
    before={k:v.detach().cpu().clone() for k,v in student.named_parameters()}
    config={**vars(args),'student_sha256':file_hash(args.source),'teacher_sha256':file_hash(args.teacher),
        'script_sha256':file_hash(__file__),'train_images':len(dataset),'student_channels':sc,'teacher_channels':tc,
        'feature_layers':LAYERS,'mosaic':0.,'mixup':0.,'copy_paste':0.,'hyp':vars(hyp),
        'data_seed_rule':'seed + 100003 * epoch, explicit sampler and workers',
        'optimizer':'SGD, single group, momentum .937, weight decay .0005, no Nesterov',
        'lr_rule':'cosine across fixed epochs, final multiplier .1',
        'loss':'batch-summed detection loss + batch_size * lambda * sum of per-scale mean feature losses',
        'primary_checkpoint':'final epoch EMA; no best-checkpoint selection',
        'EMA':'Ultralytics ModelEMA, decay .9999, tau 2000','clip_grad_norm':10.,'AMP_initial_scale':128.,
        'evaluation_epochs':[5,10,args.epochs],'temperature':args.temperature,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    start=1;elapsed=0.
    if args.resume and (out/'training_state.pt').exists():
        c=torch.load(out/'training_state.pt',map_location=f'cuda:{args.device}',weights_only=False)
        prior=json.loads((out/'config.json').read_text())
        for k in ['arm','seed','epochs','batch','lr','kd_weight','temperature','student_sha256','teacher_sha256','script_sha256']:
            assert prior[k]==config[k],f'Resume configuration changed: {k}'
        student.load_state_dict(c['student']);adapters.load_state_dict(c['adapters']);ema.ema.load_state_dict(c['ema']);ema.updates=c['ema_updates']
        optimizer.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler']);start=c['epoch']+1;elapsed=c['elapsed']
        random.setstate(c['rng_python']);np.random.set_state(c['rng_numpy']);torch.set_rng_state(c['rng_torch'].cpu());torch.cuda.set_rng_state(c['rng_cuda'].cpu(),args.device)
        del c
    else:dump(out/'config.json',config)
    # A process can stop after the atomic training checkpoint but before validation.
    completed_epoch=start-1
    if completed_epoch in [5,10,args.epochs] and not (out/f'ema_epoch{completed_epoch:02d}_metrics.json').exists():
        evaluate(args.source,{k:v.detach().cpu().clone() for k,v in ema.ema.state_dict().items()},out,args.device,args.data,f'ema_epoch{completed_epoch:02d}')
    t0=time.time();diagnostics=[]
    for epoch in range(start,args.epochs+1):
        lr=args.lr*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2)
        for g in optimizer.param_groups:g['lr']=lr
        student.train();teacher.eval();adapters.train()
        loader=make_loader(dataset,args.batch,args.workers,args.seed,epoch)
        sd=sk=0.;digests=[];norms=[];skipped=0;te=time.time()
        for bi,b in enumerate(loader):
            if bi<2:digests.append(input_digest(b))
            for k in ['img','cls','bboxes','batch_idx']:b[k]=b[k].to(args.device,non_blocking=True)
            b['img']=b['img'].float()/255.;optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                pred=student(b['img']);det,_=student.loss(b,pred);det=det.sum()
                kd=torch.zeros((),device=args.device)
                if args.arm!='control':
                    with torch.no_grad():teacher(b['img'])
                    parts=[a(fs[i]) for a,i in zip(adapters,LAYERS)]
                    if args.arm=='mse':kd=sum(F.mse_loss(a.float(),ft[i].detach().float()) for a,i in zip(parts,LAYERS))
                    else:kd=sum(cwd_loss(a,ft[i],args.temperature) for a,i in zip(parts,LAYERS))
                loss=det+b['img'].shape[0]*args.kd_weight*kd
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if epoch==1 and bi==0:
                check={'student_grad_tensors':sum(x.grad is not None for x in student.parameters()),
                    'student_nonzero_grad_tensors':sum(x.grad is not None and bool(x.grad.detach().abs().max()>0) for x in student.parameters()),
                    'adapter_grad_tensors':sum(x.grad is not None for x in adapters.parameters()),
                    'student_trainable_numel':sum(x.numel() for x in student.parameters() if x.requires_grad),
                    'detection_loss':float(det.detach()),'feature_loss':float(kd.detach()),
                    'weighted_KD_to_detection_loss':float((b['img'].shape[0]*args.kd_weight*kd/det).detach()),
                    'channel_probe_preserved_buffers':True}
                assert check['student_nonzero_grad_tensors']>0
                if args.arm!='control':assert check['adapter_grad_tensors']==6
                dump(out/'gradient_check.json',check)
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.)
            oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<oldscale:skipped+=1
            else:ema.update(student)
            sd+=float(det.detach());sk+=float(kd.detach());norms.append(float(norm))
            if bi%200==0:print(name,'epoch',epoch,'batch',bi,'/',len(loader),'det',float(det.detach()),'kd',float(kd.detach()),flush=True)
            if args.smoke_batches and bi+1>=args.smoke_batches:break
        record={'epoch':epoch,'lr':lr,'det_loss':sd/(bi+1),'feature_loss':sk/(bi+1),'seconds':time.time()-te,
            'batches':bi+1,'optimizer_steps_skipped':skipped,'median_unclipped_grad_norm':float(np.median(norms)),
            'first_two_input_sha256':digests}
        diagnostics.append(record);dump(out/f'epoch_{epoch:02d}.json',record)
        delta=[float((v.detach().cpu()-before[k]).abs().max()) for k,v in student.named_parameters()]
        assert max(delta)>0
        if args.smoke_batches:
            dump(out/'SMOKE_COMPLETE.json',{'epoch':record,'changed_parameter_tensors':sum(v>0 for v in delta),'max_delta':max(delta)})
            print('SMOKE_COMPLETE',name,flush=True);return
        state={'epoch':epoch,'student':student.state_dict(),'adapters':adapters.state_dict(),'ema':ema.ema.state_dict(),
            'ema_updates':ema.updates,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'elapsed':elapsed+time.time()-t0,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
            'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(args.device)}
        atomic_save(state,out/'training_state.pt')
        print('EPOCH_COMPLETE',name,json.dumps(record),flush=True)
        if epoch in [5,10,args.epochs]:
            metrics=evaluate(args.source,{k:v.detach().cpu().clone() for k,v in ema.ema.state_dict().items()},out,args.device,args.data,f'ema_epoch{epoch:02d}')
            print('EVALUATED',name,epoch,metrics['map50'],metrics['map50_95'],flush=True)
        del loader
    delta=[float((v.detach().cpu()-before[k]).abs().max()) for k,v in student.named_parameters()]
    fixed=next(v for k,v in student.named_parameters() if 'dfl.conv.weight' in k)
    fixed_before=next(v for k,v in before.items() if 'dfl.conv.weight' in k)
    assert torch.equal(fixed.detach().cpu(),fixed_before)
    dump(out/'parameter_delta.json',{'changed_tensors':sum(v>0 for v in delta),'max_abs_delta':max(delta),
        'fixed_dfl_preserved':True,'elapsed_seconds':elapsed+time.time()-t0})
    final=json.loads((out/f'ema_epoch{args.epochs:02d}_metrics.json').read_text())
    dump(out/'COMPLETE.json',final);print('RUN_COMPLETE',name,final['map50'],final['map50_95'],flush=True)

if __name__=='__main__':main()
