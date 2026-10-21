"""Lossless all-class prediction records and explicit post-NMS metric adapters.

This module does not perform inference, matching, threshold selection or scoring.
It rejects invalid results rather than silently fixing or removing detections.
"""
import json
import math
from pathlib import Path

INFERENCE = {'imgsz': 640, 'batch': 32, 'conf': .001, 'nms_iou': .7,
             'max_det': 300, 'half': False, 'rect': False}
TARGET_NAMES = {0: 'Pedestrian', 2: 'Car'}


def predict_kwargs(protocol, device):
    if protocol['evaluation']['inference'] != INFERENCE:
        raise ValueError('Inference settings differ from the frozen protocol')
    return {'imgsz': 640, 'batch': 32, 'conf': .001, 'iou': .7, 'max_det': 300,
            'half': False, 'rect': False, 'classes': None, 'agnostic_nms': False,
            'augment': False, 'device': device, 'verbose': False, 'save': False,
            'save_txt': False, 'stream': False}


def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if set(names) != set(range(10)) or names[0] != 'person' or names[2] != 'car':
        raise ValueError('Expected ten BDD classes with person=0 and car=2')


def image_metadata(image):
    if type(image['id']) is not int or image['id'] < 1:
        raise ValueError('Image ID must be a positive integer')
    name = image['file_name']
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Image file_name must be a basename')
    for key in ('width', 'height'):
        if type(image[key]) is not int or image[key] < 1:
            raise ValueError('Image dimensions must be positive integers')
    return {k: image[k] for k in ('id', 'file_name', 'width', 'height')}


def prediction_record(image, rows):
    """rows are original-image [x1,y1,x2,y2,score,class] after ten-class NMS."""
    metadata = image_metadata(image)
    if len(rows) > INFERENCE['max_det']:
        raise ValueError('All-class result exceeds max_det before category mapping')
    clean = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError('Each detection must contain six finite values')
        x1, y1, x2, y2, score, category = map(float, row)
        if category != int(category) or int(category) not in range(10):
            raise ValueError('Class index must be an integer from 0 through 9')
        if not 0 <= score <= 1:
            raise ValueError('Confidence must lie in [0,1]')
        if not (0 <= x1 <= x2 <= metadata['width'] and 0 <= y1 <= y2 <= metadata['height']):
            raise ValueError('Expected nonnegative-area original-image boxes already clipped by inference')
        # Boundary clipping can collapse predictions. Retain these as outputs;
        # each metric must apply its declared matching/ignore rules to them.
        clean.append([x1, y1, x2, y2, score, int(category)])
    return {'schema': 'all_class_xyxy_v1', 'image': metadata, 'detections': clean}


def encode(record):
    if record.get('schema') != 'all_class_xyxy_v1':
        raise ValueError('Unknown prediction schema')
    checked = prediction_record(record['image'], record['detections'])
    # Python JSON preserves binary64 round-trip; no deliberate rounding here.
    return (json.dumps(checked, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def decode(raw):
    record = json.loads(raw)
    checked = json.loads(encode(record))
    if checked != record:
        raise ValueError('Unexpected prediction fields or noncanonical record')
    return checked


def common_rows(record):
    checked = decode(encode(record))
    return [{'image_id': checked['image']['id'], 'category_id': c,
             'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score}
            for x1, y1, x2, y2, score, c in checked['detections'] if c in TARGET_NAMES]


def kitti_text(record):
    """Same post-NMS detections; 9 significant digits preserve float32 values."""
    checked = decode(encode(record))
    lines = []
    for x1, y1, x2, y2, score, category in checked['detections']:
        if category not in TARGET_NAMES:
            continue
        fields = [TARGET_NAMES[category], '-1', '-1', '-10',
                  *(format(v, '.9g') for v in (x1, y1, x2, y2)),
                  '-1', '-1', '-1', '-1000', '-1000', '-1000', '-10', format(score, '.9g')]
        lines.append(' '.join(fields) + '\n')
    return ''.join(lines)


def normalize_batch(results, expected_images):
    """Ensure one output per expected image, in the exact frozen input order."""
    expected_images = [image_metadata(x) for x in expected_images]
    if len({x['id'] for x in expected_images}) != len(expected_images):
        raise ValueError('Duplicate input image ID')
    if len({x['file_name'] for x in expected_images}) != len(expected_images):
        raise ValueError('Duplicate input image filename')
    if len(results) != len(expected_images):
        raise ValueError('Missing or extra result images')
    records = []
    for result, image in zip(results, expected_images):
        if Path(result.path).name != image['file_name']:
            raise ValueError('Result image order or filename mismatch')
        if tuple(result.orig_shape) != (image['height'], image['width']):
            raise ValueError('Result original-image dimensions mismatch')
        check_class_names(result.names)
        records.append(prediction_record(image, result.boxes.data.detach().cpu().tolist()))
    return records


def predict_image_chunks(model, image_dir, images, protocol, device):
    """Yield verified batches; the list-source loader can ignore batch kwargs.

    Explicitly slice the ordered input filenames before calling predict, so
    full batches contain32 images and the final partial batch is preserved.
    """
    from fixed_geometry_predictor import FixedSquarePredictor
    if model.predictor is not None and not isinstance(model.predictor, FixedSquarePredictor):
        raise ValueError('Model already has a predictor with unverified geometry')
    images = [image_metadata(x) for x in images]
    if len({x['id'] for x in images}) != len(images) or len({x['file_name'] for x in images}) != len(images):
        raise ValueError('Duplicate image ID or filename in full input manifest')
    kwargs = predict_kwargs(protocol, device)
    # Pinned8.2.103 accepts an instance here, not the predictor class.
    overrides = {**model.overrides, **{k: v for k, v in kwargs.items() if k != 'stream'},
                 'mode': 'predict', 'task': 'detect'}
    predictor = model.predictor or FixedSquarePredictor(overrides=overrides, _callbacks=model.callbacks)
    for start in range(0, len(images), INFERENCE['batch']):
        current = images[start:start + INFERENCE['batch']]
        paths = [Path(image_dir) / x['file_name'] for x in current]
        if not all(p.is_file() for p in paths):
            raise ValueError('Missing input image')
        results = model.predict(source=[str(p) for p in paths], predictor=predictor, **kwargs)
        if model.predictor.model.fp16:
            raise ValueError('Expected FP32 inference backend')
        yield normalize_batch(results, current)
