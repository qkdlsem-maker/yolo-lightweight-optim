"""Real GradScaler skip injection and exact native/counted update parity."""
import copy,json,hashlib
from pathlib import Path
import torch
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import ModelEMA
from step_accounting import OptimizerStepAccounting
R=Path('review_20261021_lr_factorial')
class Reference:
    optimizer_step=BaseTrainer.optimizer_step
class Counted(OptimizerStepAccounting,Reference):pass
torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(38217)
initial=torch.nn.Linear(8,3).cuda();x=torch.arange(32,dtype=torch.float32,device='cuda').reshape(4,8)/32
items=[]
for kind in [Reference,Counted]:
    t=kind();t.model=copy.deepcopy(initial);t.optimizer=torch.optim.SGD(t.model.parameters(),lr=.01,momentum=.937,weight_decay=.0005)
    t.scaler=torch.amp.GradScaler('cuda',init_scale=128.);t.ema=ModelEMA(t.model)
    if kind is Counted:t.install_step_accounting()
    for bad in [False,True,False]:
        with torch.autocast('cuda'): loss=t.model(x).square().sum()
        t.scaler.scale(loss).backward()
        if bad:next(t.model.parameters()).grad.fill_(float('inf'))
        t.optimizer_step()
    items.append(t)
a,b=items
for va,vb in zip(a.model.state_dict().values(),b.model.state_dict().values()):assert torch.equal(va,vb)
for va,vb in zip(a.ema.ema.state_dict().values(),b.ema.ema.state_dict().values()):assert torch.equal(va,vb)
assert a.scaler.state_dict()==b.scaler.state_dict()
for k,v in a.optimizer.state_dict()['state'].items():
    for n,t in v.items():assert torch.equal(t,b.optimizer.state_dict()['state'][k][n])
assert b.actual_step_counts=={'attempted':3,'applied':2,'skipped':1}
assert a.ema.updates==b.ema.updates==3
result={'status':'passed','actual_counts':b.actual_step_counts,'native_EMA_updates_preserved':3,
 'model_EMA_optimizer_scaler_exact_parity':True,'injected_nonfinite_step_skipped':True,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'accounting_sha256':hashlib.sha256((R/'step_accounting.py').read_bytes()).hexdigest()}
with (R/'step_accounting_validation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
