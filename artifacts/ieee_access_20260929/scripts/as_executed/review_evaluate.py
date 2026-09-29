"""Re-evaluate archived artifacts with per-class metrics and exact complexity."""
import os,sys,json,hashlib,traceback,copy,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
os.environ['YOLO_CONFIG_DIR']=str(Path.cwd()/'review_20260922/config')
import torch,numpy as np,ultralytics,torch_pruning as tp
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops,initialize_weights
from c2f_v2_utils import replace_c2f_with_c2f_v2
torch.set_num_threads(4)
out=Path('review_20260922/evaluation');out.mkdir(parents=True,exist_ok=True)
names=['yolov8n_baseline','yolov8l_baseline','yolov8s_baseline','yolov8n_kd_w03','yolov8n_kd_takd_s','yolov8n_pruned_r15_finetuned_v2','yolov8n_pruned_finetuned','yolov8n_pruned_r45_finetuned','yolov8n_pruned_r45_kd']
for name in names:
 try:
  p=Path('outputs')/name/'weights/best.pt';y=YOLO(p)
  meta={'name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'params_unfused':sum(v.numel() for v in y.model.parameters()),'gflops_thop':get_flops(y.model,640),'torch':torch.__version__,'ultralytics':ultralytics.__version__}
  if name=='yolov8n_baseline':
   a=copy.deepcopy(y.model).cpu().eval();b=copy.deepcopy(y.model).cpu().eval();replace_c2f_with_c2f_v2(b);initialize_weights(b);b.eval();torch.manual_seed(42);x=torch.rand(1,3,640,640)
   with torch.no_grad():oa=a(x)[0];ob=b(x)[0]
   meta['conversion_with_initialize_weights']={'max_abs':float((oa-ob).abs().max()),'mean_abs':float((oa-ob).abs().mean()),'allclose':torch.allclose(oa,ob,rtol=1e-4,atol=1e-5)}
  m=y.val(data='configs/bdd100k.yaml',imgsz=640,batch=32,device=0,workers=4,half=False,plots=False,save_json=False,project=str(out),name=name,exist_ok=True,verbose=False)
  meta.update({'map50':float(m.box.map50),'map50_95':float(m.box.map),'precision':float(m.box.mp),'recall':float(m.box.mr),'class_names':m.names,'class_ids':m.box.ap_class_index.tolist(),'ap50_by_class':m.box.ap50.tolist(),'ap50_95_by_class':m.box.ap.tolist(),'speed':m.speed})
  (out/(name+'.json')).write_text(json.dumps(meta,indent=2));print('COMPLETED',name,meta['map50'],meta['map50_95'],flush=True)
 except Exception:
  (out/(name+'_error.txt')).write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
print('EVALUATION_COMPLETE',flush=True)
