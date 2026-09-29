"""Fixed, resumable external inference; official local devkit does all scoring."""
import argparse,json,sys,time,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import c2f_v2_utils
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
from clean_followup_utils import sha,dump
from test_local_kitti_evaluator import read_scores,line

def format_predictions(result):
    boxes=result.boxes.data.detach().cpu().numpy()
    assert boxes.ndim==2 and boxes.shape[1]==6 and np.isfinite(boxes).all()
    rows=[]
    for x1,y1,x2,y2,confidence,category in boxes:
        if int(category) not in [0,2]:continue
        assert x2>=x1 and y2>=y1 and 0<=confidence<=1
        fields=[format(float(v),'.9g') for v in [x1,y1,x2,y2]]
        rows.append(line('Car' if int(category)==2 else 'Pedestrian',fields,format(float(confidence),'.9g')))
    return ''.join(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
    R=Path('review_20260929_clean');external=R/'external_kitti';manifest=json.loads((external/'frozen_models.json').read_text())
    assert manifest['inference_script_sha256']==sha(__file__),'Frozen inference code changed'
    entry=[r for r in manifest['models'] if r['name']==a.name];assert len(entry)==1;entry=entry[0]
    assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    out=external/'results'/a.name;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():print('ALREADY_COMPLETE',a.name);return
    config={'model':entry,'settings':manifest['inference'],'device':a.device,'frozen_manifest_sha256':sha(external/'frozen_models.json')}
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    predictions=out/'data';predictions.mkdir(exist_ok=True);batches=out/'batches';batches.mkdir(exist_ok=True)
    images=[external/'data/training/image_2'/f'{i:06d}.png' for i in range(7481)];assert all(p.exists() for p in images)
    torch.set_num_threads(2);init_seeds(0,deterministic=True);model=YOLO(entry['checkpoint'])
    assert len(model.names)==10 and model.names[0]=='person' and model.names[2]=='car',model.names
    t=time.time()
    for start in range(0,len(images),8):
        current=images[start:start+8];marker=batches/f'{start:06d}.json'
        if marker.exists():
            saved=json.loads(marker.read_text());assert list(saved)==[p.stem for p in current]
            for name,digest in saved.items():assert sha(predictions/(name+'.txt'))==digest
            continue
        results=model.predict(source=[str(p) for p in current],imgsz=640,rect=False,batch=8,half=False,conf=.001,iou=.7,max_det=300,classes=None,agnostic_nms=False,augment=False,device=a.device,verbose=False,save=False,save_txt=False,stream=False)
        assert len(results)==len(current) and not model.predictor.model.fp16
        saved={}
        for expected,result in zip(current,results):
            assert Path(result.path).stem==expected.stem
            dest=predictions/(expected.stem+'.txt');tmp=dest.with_suffix('.tmp');tmp.write_text(format_predictions(result));tmp.replace(dest);saved[expected.stem]=sha(dest)
        dump(marker,saved)
        if start%800==0:print('PREDICTIONS',a.name,start+len(current),'/7481',flush=True)
    evaluator=external/'local_evaluator/evaluate_2d_local';assert sha(evaluator)==manifest['evaluator_executable_sha256']
    subprocess.run([str(evaluator.resolve()),str(external/'data/training/label_2'),str(predictions),str(out),'7481'],check=True)
    scores=read_scores(out);aggregate=float(np.mean([scores['car'][1],scores['pedestrian'][1]]))
    h=hashlib.sha256()
    for pred in sorted(predictions.glob('*.txt')):h.update(pred.name.encode()+b'\0'+pred.read_bytes())
    result={'status':'complete','name':a.name,'group':entry['group'],'checkpoint_sha256':entry['checkpoint_sha256'],'images':7481,'metric':'KITTI native 2D AP_R40 (fraction), Car IoU>.7 and Pedestrian IoU>.5; easy/moderate/hard','class_AP_R40':scores,'primary_mean_Car_Pedestrian_moderate_AP_R40':aggregate,'predictions_sha256':h.hexdigest(),'config_sha256':sha(out/'config.json'),'seconds_this_invocation':time.time()-t,'timing_is_not_a_latency_benchmark':True,'external_public_training_partition_not_official_hidden_test':True}
    if 'arm' in entry:result.update(arm=entry['arm'],seed=entry['seed'])
    dump(out/'metrics.json',result);print('EXTERNAL_KITTI_COMPLETE',a.name,aggregate,flush=True)

if __name__=='__main__':main()
