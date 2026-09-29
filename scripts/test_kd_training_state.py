"""Regression checks for frozen checkpoints and state-mutating feature probes."""
import copy
import torch
from ultralytics import YOLO
from train_kd_feature import enable_student_training, get_channels, FEAT_LAYERS

def main():
    y=YOLO('../outputs/yolov8n_baseline/weights/best.pt')
    model=y.model.cpu().train()
    assert not any(p.requires_grad for p in model.parameters()), 'Fixture must retain frozen checkpoint flags'
    before={k:v.clone() for k,v in model.named_buffers()}
    assert get_channels(model,FEAT_LAYERS)==[64,128,256]
    assert model.training
    assert all(torch.equal(v,before[k]) for k,v in model.named_buffers()), 'Channel probe changed BatchNorm buffers'
    enable_student_training(model)
    assert model.model[-1].dfl.conv.weight.requires_grad is False
    image=torch.rand(2,3,64,64)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    first=next(model.parameters());old=first.detach().clone()
    output=model(image);loss=sum(t.square().mean() for t in output)
    optimizer.zero_grad();loss.backward()
    assert first.grad is not None and first.grad.abs().sum()>0
    optimizer.step()
    assert not torch.equal(first,old), 'Student weights did not change'
    print('PASS: frozen checkpoint is trainable, fixed DFL preserved, BatchNorm probe unchanged, optimizer updates student')

if __name__=='__main__': main()
