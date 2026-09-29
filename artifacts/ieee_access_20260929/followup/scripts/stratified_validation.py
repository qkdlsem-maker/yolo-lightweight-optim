"""Secondary evaluation on reconstructed legacy labels and fixed day/night strata.

Uses Ultralytics metric matching unchanged and records per-image sufficient
statistics for group summaries. It does not modify the training dataset.
"""
import argparse,json,hashlib,pickle,time,os,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
# Archived pruned checkpoints serialize C2f_v2 from this project module.
import c2f_v2_utils
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path.cwd()/'review_20260929/config'))
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class

class Recorder(DetectionValidator):
 def init_metrics(self,model):
  super().init_metrics(model);self.image_records=[]
 def _prepare_batch(self,si,batch):
  b=super()._prepare_batch(si,batch)
  self.image_records.append({'image':Path(batch['im_file'][si]).stem,'target_cls':b['cls'].detach().cpu().numpy(),'tp':np.zeros((0,10),dtype=bool),'conf':np.zeros(0),'pred_cls':np.zeros(0)})
  return b
 def _prepare_pred(self,pred,pbatch):
  p=super()._prepare_pred(pred,pbatch);r=self.image_records[-1]
  r.update(conf=p[:,4].detach().cpu().numpy(),pred_cls=p[:,5].detach().cpu().numpy(),tp=np.zeros((len(p),10),dtype=bool));return p
 def _process_batch(self,detections,gt_bboxes,gt_cls):
  tp=super()._process_batch(detections,gt_bboxes,gt_cls);self.image_records[-1]['tp']=tp.detach().cpu().numpy();return tp
 def get_stats(self):
  result=super().get_stats()
  stats=np.concatenate([r['target_cls'] for r in self.image_records])
  assert np.array_equal(np.bincount(stats.astype(int),minlength=10),self.nt_per_class)
  return result

def aggregate(records):
 values={k:np.concatenate([r[k] for r in records]) for k in ['tp','conf','pred_cls','target_cls']}
 result=ap_per_class(**values,plot=False)
 ap=result[5];ids=result[6].astype(int)
 class_ap={str(i):{'AP50':float(v[0]),'AP50_95':float(v.mean())} for i,v in zip(ids,ap)}
 road=[class_ap[str(i)] for i in range(9) if str(i) in class_ap]
 counts=np.bincount(values['target_cls'].astype(int),minlength=10)
 return {'images':len(records),'instances_by_class':counts.tolist(),'class_metrics':class_ap,'AP50_present_classes':float(ap[:,0].mean()),'AP50_95_present_classes':float(ap.mean()),'road9_AP50':float(np.mean([x['AP50'] for x in road])) if len(road)==9 else None,'road9_AP50_95':float(np.mean([x['AP50_95'] for x in road])) if len(road)==9 else None}

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--batch',type=int,default=8)
 p.add_argument('--data',default='review_20260929/verified_legacy/data.yaml');p.add_argument('--attributes',default='review_20260929/legacy_val_attributes.json');p.add_argument('--out-root',default='review_20260929/stratified');p.add_argument('--annotation-set',default='reconstructed BDD100K 2018 JSON validation mirror');a=p.parse_args()
 torch.set_num_threads(2);out=Path(a.out_root)/a.name
 if (out/'metrics.json').exists():print('ALREADY_COMPLETE',a.name);return
 out.mkdir(parents=True,exist_ok=True);attrs=json.loads(Path(a.attributes).read_text());t=time.time()
 model=YOLO(a.checkpoint);holder={}
 def factory(*args,**kwargs):
  validator=Recorder(*args,**kwargs);holder['validator']=validator;return validator
 m=model.val(validator=factory,data=a.data,imgsz=640,batch=a.batch,device=a.device,workers=2,half=False,plots=False,verbose=False,project=str(out),name='validation')
 records=holder['validator'].image_records
 assert len(records)==10000 and {r['image'] for r in records}==set(attrs)
 overall=aggregate(records)
 assert abs(overall['AP50_present_classes']-float(m.box.map50))<1e-12
 assert abs(overall['AP50_95_present_classes']-float(m.box.map))<1e-12
 strata={}
 for value in sorted({x['timeofday'] for x in attrs.values()}):
  selected=[r for r in records if attrs[r['image']]['timeofday']==value]
  strata[value]=aggregate(selected)
 pred_counts=[len(r['conf']) for r in records];gt_counts=[len(r['target_cls']) for r in records]
 values={k:np.concatenate([r[k] for r in records]) for k in ['tp','conf','pred_cls','target_cls']}
 np.savez_compressed(out/'image_statistics.npz',**values,pred_offsets=np.cumsum([0]+pred_counts),gt_offsets=np.cumsum([0]+gt_counts),images=np.array([r['image'] for r in records]))
 result={'checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'annotation_set':a.annotation_set,'batch':a.batch,'imgsz':640,'fp16':False,'overall':overall,'timeofday':strata,'all_10_classes_for_overall':True,'strata_road9':'Fixed nine road-object classes; train excluded because of extremely sparse support; null if any of the nine classes is absent','statistics_reproduce_validator_AP':True,'seconds':time.time()-t,'timing_is_not_a_latency_benchmark':True}
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print('SECONDARY_EVALUATION_COMPLETE',a.name,overall['AP50_present_classes'],overall['AP50_95_present_classes'],flush=True)

if __name__=='__main__':main()
