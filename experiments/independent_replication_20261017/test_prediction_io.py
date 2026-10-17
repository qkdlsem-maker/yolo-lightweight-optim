"""Synthetic known-answer and fail-closed prediction serialization tests."""
import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from prediction_io import (INFERENCE, check_class_names, common_rows, decode, encode,
                           kitti_text, normalize_batch, predict_kwargs, prediction_record)

ROOT = Path(__file__).resolve().parent
checks = []
image = {'id': 1, 'file_name': '000000.png', 'width': 1242, 'height': 375}
names = {0: 'person', 1: 'rider', 2: 'car', **{i: f'other{i}' for i in range(3, 10)}}
rows = [[1.25, 2.5, 100.75, 150.5, .75, 2], [3, 4, 60, 120, .25, 0], [0, 0, 1242, 375, .9, 1]]
original = copy.deepcopy(rows)
record = prediction_record(image, rows)
assert rows == original and len(record['detections']) == 3
checks.append('All ten-class outputs retained without mutating inputs')
assert decode(encode(record)) == record
checks.append('All-class JSON round trip retains coordinates and confidences')
assert common_rows(record) == [
    {'image_id': 1, 'category_id': 2, 'bbox': [1.25, 2.5, 99.5, 148.], 'score': .75},
    {'image_id': 1, 'category_id': 0, 'bbox': [3., 4., 57., 116.], 'score': .25}]
checks.append('Common mapping happens after NMS; xyxy-to-xywh uses no plus one')
native = [x.split() for x in kitti_text(record).splitlines()]
assert [x[0] for x in native] == ['Car', 'Pedestrian'] and all(len(x) == 16 for x in native)
assert list(map(float, native[0][4:8])) == rows[0][:4] and float(native[0][-1]) == .75
checks.append('Native KITTI mapping preserves order, coordinates and confidence')
empty = prediction_record(image, [])
assert decode(encode(empty)) == empty and common_rows(empty) == [] and kitti_text(empty) == ''
checks.append('Empty prediction image retained with explicit metadata')
assert len(prediction_record(image, [rows[2]] * 300)['detections']) == 300
checks.append('All-class cap of exactly 300 accepted before target mapping')
f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
precision = [f32(v) for v in (1.23456789, 2.34567891, 100.9876543, 150.8765432, .123456789)]
precise = prediction_record(image, [precision + [2]])
assert decode(encode(precise))['detections'][0][:5] == precision
parts = kitti_text(precise).split()
assert [f32(float(v)) for v in parts[4:8] + parts[-1:]] == precision
checks.append('Nine-digit native text and JSON round trip recover float32 test values exactly')
class FakeTensor:
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return rows
result = SimpleNamespace(path='/synthetic/000000.png', orig_shape=(375, 1242), names=names,
                         boxes=SimpleNamespace(data=FakeTensor()))
assert normalize_batch([result], [image]) == [record]
checks.append('Batch filename, shape and class mapping validated before serialization')
kwargs = predict_kwargs({'evaluation': {'inference': INFERENCE}}, 'cpu')
assert kwargs['classes'] is None and kwargs['batch'] == 32 and kwargs['rect'] is False
assert kwargs['half'] is False and kwargs['agnostic_nms'] is False and kwargs['augment'] is False
checks.append('Frozen inference options include all-class class-aware NMS and FP32 square input')

def reject(name, operation):
    try: operation()
    except ValueError: checks.append(name)
    else: raise AssertionError('Silently accepted: ' + name)

reject('Class filtering cannot hide an exceeded all-class cap', lambda: prediction_record(image, [rows[2]] * 301))
reject('Fractional class indices rejected', lambda: prediction_record(image, [[1, 2, 3, 4, .5, 2.5]]))
reject('Out-of-range class indices rejected', lambda: prediction_record(image, [[1, 2, 3, 4, .5, 10]]))
reject('NaN confidence rejected', lambda: prediction_record(image, [[1, 2, 3, 4, float('nan'), 2]]))
reject('Infinite coordinate rejected', lambda: prediction_record(image, [[1, 2, float('inf'), 4, .5, 2]]))
reject('Confidence above one rejected', lambda: prediction_record(image, [[1, 2, 3, 4, 1.1, 2]]))
collapsed = prediction_record(image, [[1, 2, 1, 4, .5, 2]])
assert common_rows(collapsed)[0]['bbox'] == [1., 2., 0., 2.]
assert len(kitti_text(collapsed).splitlines()) == 1 and decode(encode(collapsed)) == collapsed
checks.append('Zero-area clipped predictions retained in every serialized view')
reject('Reversed boxes rejected', lambda: prediction_record(image, [[3, 2, 1, 4, .5, 2]]))
reject('Unclipped box rejected', lambda: prediction_record(image, [[-1, 2, 3, 4, .5, 2]]))
reject('Wrong result image rejected', lambda: normalize_batch([result], [dict(image, file_name='other.png')]))
reject('Missing result image rejected', lambda: normalize_batch([], [image]))
reject('Wrong source dimensions rejected', lambda: normalize_batch([result], [dict(image, height=374)]))
reject('Duplicate image IDs rejected', lambda: normalize_batch([result, result], [image, image]))
reject('Path traversal filename rejected', lambda: prediction_record(dict(image, file_name='../x.png'), []))
reject('Wrong class-name mapping rejected', lambda: check_class_names(dict(names, **{'0': 'person'})))
reject('Changed inference setting rejected', lambda: predict_kwargs({'evaluation': {'inference': dict(INFERENCE, rect=True)}}, 'cpu'))
reject('Unknown prediction schema rejected', lambda: encode(dict(record, schema='unknown')))
evidence = {'status': 'passed', 'checks': checks, 'real_images_or_model_scores_used': False,
            'files_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                             for n in ['prediction_io.py', 'test_prediction_io.py']}}
(ROOT / 'prediction_io_validation.json').write_bytes((json.dumps(evidence, indent=2) + '\n').encode())
print(json.dumps({'status': 'passed', 'checks': len(checks)}))
