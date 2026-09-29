from pathlib import Path
import sys,json,copy
sys.path.insert(0,'scripts')
import torch
from ultralytics import YOLO
from train_kd_feature import FeatureAdapters,FEAT_LAYERS,register_feature_hooks,get_channels
from c2f_v2_utils import replace_c2f_with_c2f_v2
torch.set_num_threads(4)
base=YOLO('outputs/yolov8n_baseline/weights/best.pt').model
out={}
for name in ['yolov8n_kd_w03','yolov8n_kd_takd_s']:
 state=torch.load(Path('outputs')/name/'weights/best_student_state_dict.pt',map_location='cpu',weights_only=True)
 params=dict(base.named_parameters());unchanged=[];changed=[]
 for k,p in params.items():
  (unchanged if torch.equal(p,state[k]) else changed).append(k)
 buffers={k:float((v.float()-state[k].float()).abs().max()) for k,v in base.named_buffers() if k in state}
 out[name]={'parameter_tensors':len(params),'unchanged_parameter_tensors':len(unchanged),'changed_parameters':changed,'max_parameter_difference':max(float((p-state[k]).abs().max()) for k,p in params.items()),'changed_buffer_count':sum(v!=0 for v in buffers.values()),'max_buffer_difference':max(buffers.values())}
student=copy.deepcopy(base).cuda().train()
teacher=YOLO('outputs/yolov8l_baseline/weights/best.pt').model.cuda().eval()
for p in teacher.parameters():p.requires_grad_(False)
sc=get_channels(student,FEAT_LAYERS);tc=get_channels(teacher,FEAT_LAYERS)
adapters=FeatureAdapters(sc,tc).cuda();fs={};ft={}
register_feature_hooks(student,FEAT_LAYERS,fs);register_feature_hooks(teacher,FEAT_LAYERS,ft)
x=torch.rand(2,3,640,640,device='cuda')
student(x)
with torch.no_grad():teacher(x)
loss=sum(torch.nn.functional.mse_loss(a,ft[i]) for a,i in zip(adapters([fs[i] for i in FEAT_LAYERS]),FEAT_LAYERS))
loss.backward()
out['original_gradient_probe']={'student_trainable_numel':sum(p.numel() for p in student.parameters() if p.requires_grad),'student_grad_tensors':sum(p.grad is not None for p in student.parameters()),'adapter_grad_tensors':sum(p.grad is not None for p in adapters.parameters()),'loss':float(loss),'student_channels':sc,'teacher_channels':tc}
student.zero_grad();adapters.zero_grad()
for n,p in student.named_parameters(): p.requires_grad_('dfl.conv.weight' not in n)
student(x);loss=sum(torch.nn.functional.mse_loss(a,ft[i]) for a,i in zip(adapters([fs[i] for i in FEAT_LAYERS]),FEAT_LAYERS));loss.backward()
out['corrected_gradient_probe']={'student_trainable_numel':sum(p.numel() for p in student.parameters() if p.requires_grad),'student_grad_tensors':sum(p.grad is not None for p in student.parameters()),'adapter_grad_tensors':sum(p.grad is not None for p in adapters.parameters())}
torch.manual_seed(42)
m1=copy.deepcopy(base).cpu().eval();m2=copy.deepcopy(base).cpu().eval();replace_c2f_with_c2f_v2(m2);m2.eval()
xx=torch.rand(1,3,640,640)
with torch.no_grad():a=m1(xx)[0];b=m2(xx)[0]
out['c2f_conversion']={'max_abs_diff':float((a-b).abs().max()),'mean_abs_diff':float((a-b).abs().mean()),'allclose_rtol1e-4_atol1e-5':torch.allclose(a,b,rtol=1e-4,atol=1e-5)}
Path('review_20260922/kd_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
