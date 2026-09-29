"""Interleaved, model-only CUDA-event timing; block raw data are retained."""
import os,sys,json,random,time,subprocess,statistics
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
os.environ['YOLO_CONFIG_DIR']=str(Path.cwd()/'review_20260922/config')
import torch
from ultralytics import YOLO
import c2f_v2_utils
torch.set_num_threads(1);torch.cuda.set_device(0);torch.backends.cudnn.benchmark=True
names=['yolov8n_baseline','yolov8l_baseline','yolov8n_pruned_r15_finetuned_v2','yolov8n_pruned_finetuned','yolov8n_pruned_r45_finetuned']
models={};records=[];torch.manual_seed(0)
for name in names:
 for precision in ['fp32','fp16']:
  m=YOLO(f'outputs/{name}/weights/best.pt').model.cuda().eval();m.fuse(verbose=False);m=m.half() if precision=='fp16' else m.float();x=torch.rand(1,3,640,640,device='cuda').to(next(m.parameters()).dtype)
  models[name+'_'+precision]=(m,x)
config={'batch':1,'shape':[1,3,640,640],'device':0,'warmup':100,'blocks':10,'iterations_per_block':200,'seed':20260922,'model_boundary':'device-resident tensor through fused PyTorch model; excludes decoding, transfer, preprocessing, NMS','gpu_start':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,power.draw,clocks.sm,temperature.gpu','--format=csv']).decode(),'gpu_processes_start':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']).decode()}
with torch.inference_mode():
 for key,(m,x) in models.items():
  for _ in range(100):m(x)
  torch.cuda.synchronize()
 rng=random.Random(20260922)
 for b in range(10):
  keys=list(models);rng.shuffle(keys)
  for key in keys:
   m,x=models[key];a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();a.record()
   for _ in range(200):m(x)
   z.record();torch.cuda.synchronize();ms=a.elapsed_time(z)/200;records.append({'block':b,'model':key,'mean_ms':ms,'fps':1000/ms});print(b,key,ms,flush=True)
summaries={}
for key in models:
 v=[r['mean_ms'] for r in records if r['model']==key];summaries[key]={'mean_ms':statistics.mean(v),'sd_ms':statistics.stdev(v),'median_ms':statistics.median(v),'fps_from_mean_ms':1000/statistics.mean(v),'block_cv_pct':100*statistics.stdev(v)/statistics.mean(v)}
Path('review_20260922/benchmark.json').write_text(json.dumps({'config':config,'raw_blocks':records,'summary':summaries},indent=2));print('BENCHMARK_COMPLETE',flush=True)
