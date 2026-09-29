"""Fixed-budget, paired-seed warm-start experiment. Never overwrites old outputs."""
import argparse,os,sys,json,time,random,csv,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
os.environ['YOLO_CONFIG_DIR']=str(Path.cwd()/'review_20260922/config')
import torch,numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset,build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import init_seeds
from train_kd_feature import FeatureAdapters,FEAT_LAYERS,register_feature_hooks,get_channels

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--arm',choices=['control','kd'],required=True);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--epochs',type=int,default=10);args=p.parse_args()
 torch.set_num_threads(4);torch.cuda.set_device(args.device)
 for seed in args.seeds:
  name=f'{args.arm}_seed{seed}_e{args.epochs}';out=Path('review_20260922/controlled')/name;out.mkdir(parents=True,exist_ok=False)
  init_seeds(seed,deterministic=True)
  source='outputs/yolov8n_baseline/weights/best.pt';teacher_path='outputs/yolov8l_baseline/weights/best.pt';sy=YOLO(source);student=sy.model.to(args.device);teacher=YOLO(teacher_path).model.to(args.device).eval()
  for x in teacher.parameters():x.requires_grad_(False)
  for n,x in student.named_parameters():x.requires_grad_('dfl.conv.weight' not in n)
  hyp=get_cfg(DEFAULT_CFG);hyp.imgsz=640;hyp.seed=seed;hyp.deterministic=True
  student.args=hyp;teacher.args=hyp
  student.eval();sc=get_channels(student,FEAT_LAYERS);tc=get_channels(teacher,FEAT_LAYERS);student.train()
  adapters=FeatureAdapters(sc,tc).to(args.device);fs={};ft={};register_feature_hooks(student,FEAT_LAYERS,fs);register_feature_hooks(teacher,FEAT_LAYERS,ft)
  opt=torch.optim.SGD(list(student.parameters())+list(adapters.parameters()),lr=.001,momentum=.937,weight_decay=.0005)
  scaler=torch.amp.GradScaler('cuda')
  # Reset after identical teacher/adapter setup so both arms receive the same data RNG seed.
  init_seeds(seed,deterministic=True)
  data=check_det_dataset('configs/bdd100k.yaml');ds=build_yolo_dataset(hyp,data['train'],64,data,mode='train',rect=False);loader=build_dataloader(ds,64,4,shuffle=True)
  conf={'arm':args.arm,'seed':seed,'epochs':args.epochs,'batch':64,'lr':.001,'optimizer':'SGD','momentum':.937,'weight_decay':.0005,'kd_weight':.3 if args.arm=='kd' else 0.,'student_source':source,'student_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'teacher':teacher_path,'feature_layers':FEAT_LAYERS,'student_channels':sc,'teacher_channels':tc,'checkpoint_selection':'fixed final epoch','loss_scaling':'Ultralytics detection loss summed over batch plus weight times three per-element mean MSE losses','train_images':len(ds),'deterministic':True}
  (out/'config.json').write_text(json.dumps(conf,indent=2));before={n:v.detach().cpu().clone() for n,v in student.named_parameters()};t0=time.time()
  with (out/'train.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['epoch','det_loss','kd_loss','seconds'])
   for epoch in range(1,args.epochs+1):
    student.train();te=time.time();sd=sk=0.
    for bi,b in enumerate(loader):
     for k in ['img','cls','bboxes','batch_idx']:b[k]=b[k].to(args.device,non_blocking=True)
     b['img']=b['img'].float()/255.;opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda'):
      preds=student(b['img']);det,_=student.loss(b,preds);det=det.sum()
      kd=torch.zeros((),device=args.device)
      if args.arm=='kd':
       with torch.no_grad():teacher(b['img'])
       kd=sum(torch.nn.functional.mse_loss(a,ft[i].detach()) for a,i in zip(adapters([fs[i] for i in FEAT_LAYERS]),FEAT_LAYERS))
      loss=det+conf['kd_weight']*kd
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
     scaler.scale(loss).backward()
     if epoch==1 and bi==0:
      (out/'gradient_check.json').write_text(json.dumps({'student_grad_tensors':sum(v.grad is not None for v in student.parameters()),'student_trainable_numel':sum(v.numel() for v in student.parameters() if v.requires_grad),'adapter_grad_tensors':sum(v.grad is not None for v in adapters.parameters())},indent=2))
      assert sum(v.grad is not None for v in student.parameters())>0
     scaler.step(opt);scaler.update();sd+=float(det.detach());sk+=float(kd.detach())
     if bi%200==0:print(name,epoch,bi,len(loader),'loss',float(loss.detach()),flush=True)
    w.writerow([epoch,sd/len(loader),sk/len(loader),time.time()-te]);f.flush();print('EPOCH',name,epoch,sd/len(loader),sk/len(loader),time.time()-te,flush=True)
  delta={n:float((v.detach().cpu()-before[n]).abs().max()) for n,v in student.named_parameters()};(out/'parameter_delta.json').write_text(json.dumps({'changed_tensors':sum(x>0 for x in delta.values()),'max_abs_delta':max(delta.values()),'elapsed_seconds':time.time()-t0},indent=2));assert max(delta.values())>0
  fresh=YOLO(source);fresh.model.load_state_dict({k:v.detach().cpu().clone() for k,v in student.state_dict().items()});fresh.save(str(out/'final.pt'))
  del teacher,student,sy,adapters,opt,fs,ft,loader,ds;torch.cuda.empty_cache()
  metrics=fresh.val(data='configs/bdd100k.yaml',imgsz=640,batch=32,device=args.device,workers=4,half=False,plots=False,project=str(out),name='validation',verbose=False)
  (out/'metrics.json').write_text(json.dumps({'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'classes':metrics.names,'ap50':metrics.box.ap50.tolist(),'ap50_95':metrics.box.ap.tolist()},indent=2));print('RUN_COMPLETE',name,metrics.box.map50,metrics.box.map,flush=True)
 print('ALL_COMPLETE',args.arm,flush=True)
if __name__=='__main__':main()
