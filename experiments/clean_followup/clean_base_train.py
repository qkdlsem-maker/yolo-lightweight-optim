"""Fresh fixed-epoch BDD100K training on the verified-source label subset."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
from ultralytics import YOLO

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['nano','large'],required=True);parser.add_argument('--device',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=Path('review_20260929_clean');source=root/'pretrained'/('yolov8n.pt' if args.model=='nano' else 'yolov8l.pt')
    suffix='_smoke' if args.smoke else '';run=root/'base'/(args.model+suffix);record=root/'base_records'/(args.model+suffix);record.mkdir(parents=True,exist_ok=True)
    if (record/'COMPLETE.json').exists():print('ALREADY_COMPLETE',args.model,suffix);return
    epochs=1 if args.smoke else 100
    config={'source':str(source),'source_sha256':sha(source),'data':'review_20260929_clean/clean_bdd.yaml','data_sha256':sha(root/'clean_bdd.yaml'),'script_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.md'),
            'model':args.model,'epochs':epochs,'batch':64 if args.model=='nano' else 16,'nbs':64,'device':args.device,'imgsz':640,'seed':0,'optimizer':'SGD','lr0':.01,'lrf':.01,'cos_lr':False,'momentum':.937,'weight_decay':.0005,'warmup_epochs':3.,'patience':0,'close_mosaic':10,'amp':True,'workers':8,'fraction':.002 if args.smoke else 1.,'endpoint':'final epoch EMA in last.pt, never best.pt','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if (record/'config.json').exists():
        prior=json.loads((record/'config.json').read_text())
        for key in config.keys()-{'created_utc'}:assert prior[key]==config[key],key
        config=prior
        assert args.resume,'Existing incomplete run requires explicit resume'
    else:dump(record/'config.json',config)
    torch.set_num_threads(4)
    def progress(trainer):
        loss=trainer.tloss.detach().cpu()
        if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite native training loss')
        dump(record/'progress.json',{'epoch':trainer.epoch+1,'epochs':epochs,'loss_items':loss.tolist(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    checkpoint=run/'weights/last.pt'
    if args.resume:
        assert checkpoint.exists();model=YOLO(str(checkpoint));model.add_callback('on_fit_epoch_end',progress)
        model.train(resume=True,device=args.device)
    else:
        assert not run.exists(),run
        model=YOLO(str(source));model.add_callback('on_fit_epoch_end',progress)
        model.train(data=config['data'],epochs=epochs,batch=config['batch'],nbs=64,device=args.device,imgsz=640,seed=0,deterministic=True,
                    optimizer='SGD',lr0=.01,lrf=.01,cos_lr=False,momentum=.937,weight_decay=.0005,warmup_epochs=3.,patience=0,
                    close_mosaic=10,amp=True,workers=8,fraction=config['fraction'],pretrained=True,project=str(root/'base'),name=args.model+suffix,
                    exist_ok=False,save=True,save_period=10,cache=False,plots=False,verbose=False,val=True)
    rows=list(csv.DictReader((run/'results.csv').open()));assert len(rows)==epochs
    saved=YOLO(str(checkpoint)).model
    assert saved.model[-1].nc==10
    bad=[name for name,v in saved.state_dict().items() if not torch.isfinite(v).all()];assert not bad,bad
    dump(record/'COMPLETE.json',{'status':'complete','epochs':epochs,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'source_sha256':config['source_sha256'],'data_manifest_sha256':sha(root/'clean_data_manifest.json'),'finite_state':True,'last_row':rows[-1],'endpoint':'last epoch EMA; best.pt is not used as a continuation input'})
    print('CLEAN_BASE_COMPLETE',args.model,suffix,flush=True)

if __name__=='__main__':main()
