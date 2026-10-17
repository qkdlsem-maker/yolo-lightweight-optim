"""Verify native standalone validation geometry on a generated 33-image set.

Labels and images are artificial software fixtures. Accuracy/latency numbers
from this invocation are deliberately not retained as experimental results.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import init_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not replace prior verification'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert importlib.metadata.version('ultralytics') == '8.2.103'
    checkpoint_hash = sha(args.checkpoint)
    protocol = json.loads(args.protocol.read_bytes())
    expected = {'imgsz': 640, 'batch': 32, 'conf': .001, 'nms_iou': .7,
                'max_det': 300, 'half': False, 'rect': False}
    assert protocol['evaluation']['inference'] == expected
    torch.set_num_threads(2)
    init_seeds(0, deterministic=True)
    model = YOLO(str(args.checkpoint))
    observed, initialization_probes, loader = [], [], {}
    validation_started = False
    def capture(module, inputs):
        t = inputs[0]
        trace = {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': t.device.type}
        (observed if validation_started else initialization_probes).append(trace)
    handle = model.model.register_forward_pre_hook(capture)
    def val_start(validator):
        nonlocal validation_started
        validation_started = True
        loader.update(images=len(validator.dataloader.dataset),
                      batch=validator.dataloader.batch_size,
                      rectangle=bool(validator.dataloader.dataset.rect),
                      half=bool(validator.args.half), training_context=bool(validator.training))
    model.add_callback('on_val_start', val_start)
    with tempfile.TemporaryDirectory(prefix='synthetic_native_val_') as temporary:
        root = Path(temporary)
        images = root / 'images/val'; labels = root / 'labels/val'
        images.mkdir(parents=True); labels.mkdir(parents=True)
        for i in range(33):
            # Same native shape as BDD: homogeneous batches must still be square.
            pixels = np.full((720, 1280, 3), (i % 3) * 127, dtype=np.uint8)
            Image.fromarray(pixels).save(images / f'synthetic_{i:03d}.png')
            (labels / f'synthetic_{i:03d}.txt').write_text(f'{2 if i % 2 == 0 else 0} 0.5 0.5 0.1 0.1\n')
        data = root / 'synthetic.yaml'
        data.write_text(yaml.safe_dump({'path': str(root), 'train': 'images/val',
                                       'val': 'images/val', 'names': model.names}))
        model.val(data=str(data), imgsz=640, batch=32, device='cpu', workers=0,
                  half=False, rect=False, conf=.001, iou=.7, max_det=300, plots=False,
                  verbose=False, project=str(root / 'results'), name='geometry',
                  save_json=False, save_txt=False, save_hybrid=False)
    handle.remove()
    assert loader == {'images': 33, 'batch': 32, 'rectangle': False,
                      'half': False, 'training_context': False}, loader
    assert [x['shape'] for x in observed] == [[32, 3, 640, 640], [1, 3, 640, 640]], observed
    assert all(x['dtype'] == 'torch.float32' and x['device'] == 'cpu' for x in observed)
    assert sha(args.checkpoint) == checkpoint_hash
    record = {'status': 'passed', 'scope': 'Native standalone validation on artificial 33-image/33-label fixtures only',
              'real_BDD_KITTI_data_used': False, 'synthetic_accuracy_or_latency_reported_as_research': False,
              'input_image_dimensions': [1280, 720], 'inference': expected, 'dataloader': loader,
              'actual_validation_forward_inputs': observed,
              'before_on_val_start_model_initialization_probes': initialization_probes,
              'checkpoint_unchanged': True,
              'preflight_smoke_checkpoint_sha256': checkpoint_hash, 'protocol_sha256': sha(args.protocol),
              'script_sha256': sha(Path(__file__)),
              'native_source_text_sha256': {name: hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
                 for name, value in [('build_yolo_dataset', build_yolo_dataset),
                                     ('DetectionValidator', DetectionValidator)]},
              'runtime': {n: importlib.metadata.version(n) for n in ('ultralytics', 'torch', 'numpy')},
              'limits': 'This verifies software geometry on CPU, not all real-data inputs or GPU numeric equivalence. Training-context native validation can use different precision/geometry. All frozen endpoints still require standalone evaluation and actual input checks.'}
    args.output.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
