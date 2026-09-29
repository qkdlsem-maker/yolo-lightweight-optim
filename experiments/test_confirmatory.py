import torch
import torch.nn.functional as F
from confirmatory_kd import cwd_loss

torch.manual_seed(71)
s=torch.randn(2,3,7,9,requires_grad=True);t=torch.randn_like(s,requires_grad=True);tau=4.
actual=cwd_loss(s,t,tau)
ts=F.softmax(t.detach().reshape(-1,63)/tau,dim=1)
reference=(ts*(F.log_softmax(t.detach().reshape(-1,63)/tau,dim=1)-F.log_softmax(s.reshape(-1,63)/tau,dim=1))).sum()*tau*tau/6
assert torch.allclose(actual,reference,atol=1e-6)
a=torch.autograd.grad(actual,s,retain_graph=True)[0];b=torch.autograd.grad(reference,s)[0]
assert torch.allclose(a,b,atol=1e-6)
assert cwd_loss(t,t,tau).abs()<1e-6
assert torch.allclose(cwd_loss(s+torch.randn(2,3,1,1),t,tau),actual,atol=1e-5)
actual.backward();assert s.grad.abs().sum()>0 and t.grad is None
print('PASS: CWD reference values and gradients, identity, channel-shift invariance, and detached teacher')
