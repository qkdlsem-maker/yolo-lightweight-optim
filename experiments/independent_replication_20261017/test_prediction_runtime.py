"""CPU-only full/partial-batch inference on 33 generated, unlabeled images.

Use a preflight smoke checkpoint. No BDD/KITTI image, label or accuracy score is
read; this checks actual tensor geometry, precision and serialized image IDs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
from prediction_io import encode, decode, kitti_text, predict_image_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing validation evidence'
    assert importlib.metadata.version('ultralytics') == '8.2.103'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = sha(args.checkpoint)
    protocol = json.loads(args.protocol.read_bytes())
    torch.set_num_threads(2)
    init_seeds(0, deterministic=True)
    model = YOLO(str(args.checkpoint))
    observed = []
    def capture(module, inputs):
        tensor = inputs[0]
        observed.append({'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'device': tensor.device.type})
    handle = model.model.register_forward_pre_hook(capture)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='synthetic_prediction_io_') as folder:
        files, images = [], []
        for i in range(33):
            width, height = [(1280, 720), (1242, 375), (1241, 376)][i % 3]
            name = f'synthetic_{i:03d}.png'
            path = Path(folder) / name
            pixels = np.full((height, width, 3), (i % 3) * 127, dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            files.append(str(path))
            images.append({'id': i + 1, 'file_name': name, 'width': width, 'height': height})
        batches = list(predict_image_chunks(model, folder, images, protocol, 'cpu'))
        assert [len(b) for b in batches] == [32, 1]
        records = [r for batch in batches for r in batch]
        assert len(records) == 33 and not model.predictor.model.fp16
        assert all(t['shape'][1:] == [3, 640, 640] and t['dtype'] == 'torch.float32'
                   and t['device'] == 'cpu' for t in observed), observed
        assert [t['shape'][0] for t in observed] == [32, 1], observed
        for record in records:
            assert decode(encode(record)) == record
            assert all(len(line.split()) == 16 for line in kitti_text(record).splitlines())
    handle.remove()
    assert sha(args.checkpoint) == before
    evidence = {'status': 'passed', 'scope': 'Synthetic image inference and IO only, not detector accuracy or latency measurement',
                'images': 33, 'actual_forward_inputs': observed, 'inference': protocol['evaluation']['inference'],
                'all_class_agnostic_NMS': False, 'class_filter_before_NMS': None,
                'real_data_or_performance_scores_used': False, 'checkpoint_unchanged': True,
                'total_predictions_retained': sum(len(r['detections']) for r in records),
                'zero_area_predictions_retained': sum(x1 == x2 or y1 == y2 for r in records
                    for x1, y1, x2, y2, score, category in r['detections']),
                'preflight_smoke_checkpoint_sha256': before, 'protocol_sha256': sha(args.protocol),
                'elapsed_seconds_not_a_latency_benchmark': time.monotonic() - started,
                'runtime': {n: importlib.metadata.version(n) for n in ('torch', 'ultralytics', 'numpy')},
                'source_sha256': {n: sha(Path(__file__).parent / n) for n in
                                 ('prediction_io.py', 'test_prediction_runtime.py', 'fixed_geometry_predictor.py')}}
    args.output.write_bytes((json.dumps(evidence, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
