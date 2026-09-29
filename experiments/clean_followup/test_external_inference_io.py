"""Check class mapping/coordinate serialization and FP32 prediction on a blank image.

No KITTI image, annotation or model score is used for this software smoke test.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics import YOLO
from external_kitti_inference import format_predictions
from clean_followup_utils import dump
R=Path('review_20260929_clean')
sample=torch.tensor([[1.25,2.5,100.75,150.5,.75,2],[3,4,60,120,.25,0],[1,2,50,60,.9,1]],dtype=torch.float32)
text=format_predictions(SimpleNamespace(boxes=SimpleNamespace(data=sample)))
rows=[r.split() for r in text.splitlines()];assert len(rows)==2
assert rows[0][0]=='Car' and rows[1][0]=='Pedestrian' and all(len(r)==16 for r in rows)
assert list(map(float,rows[0][4:8]))==[1.25,2.5,100.75,150.5]
assert float(rows[0][-1])==.75 and float(rows[1][-1])==.25
torch.set_num_threads(2)
model=YOLO(str(R/'base/nano_smoke/weights/last.pt'))
result=model.predict(np.zeros((376,1242,3),dtype=np.uint8),imgsz=640,rect=False,batch=8,half=False,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False,save_txt=False)
assert len(result)==1 and not model.predictor.model.fp16
format_predictions(result[0])
dump(R/'external_kitti/inference_io_verification.json',{'passed':True,'mapping':'BDD2 to Car, BDD0 to Pedestrian; all other classes dropped after NMS','coordinate_and_confidence_serialization_verified':True,'blank_synthetic_image_FP32_prediction_verified':True,'real_external_data_used':False})
print('EXTERNAL_INFERENCE_IO_VERIFIED',flush=True)
