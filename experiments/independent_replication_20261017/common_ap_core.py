"""Shared two-class AP core; dataset-specific annotation adapters are separate.

Uses the pinned COCO evaluator, not the Ultralytics trapezoidal AP or KITTI R40.
This module alone does not establish annotation comparability between datasets.
"""
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VERSION = '2.0.11'
CLASSES = {0: 'person', 2: 'car'}
IOU = np.linspace(.50, .95, 10)
RECALL = np.linspace(0., 1., 101)
MAX_DET = 300


def runtime_record():
    version = importlib.metadata.version('pycocotools')
    if version != VERSION:
        raise ValueError(f'Requires pycocotools=={VERSION}, found {version}')
    import pycocotools.cocoeval as implementation
    return {'pycocotools_version': version, 'numpy_version': np.__version__,
            'cocoeval_python_sha256': hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()}


def _bbox(row):
    box = row['bbox']
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError('A bbox must contain four finite xywh values')
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError('Zero/negative box areas must be explicitly handled by the data adapter')
    return float(box[2] * box[3])


def evaluate(images, ground_truth, predictions, include_matches=False):
    """Evaluate normalized COCO-format rows without silently filtering anything.

    images: id, width, height; ground_truth: id, image_id, category_id, bbox,
    area and explicit iscrowd; predictions: image_id, category_id, bbox, score.
    Require at least one ordinary target of each class. All image IDs, including
    empty images, must be supplied. Crowds use standard COCO ignore behavior;
    interpreting dataset ignore annotations is NOT delegated to this function.
    """
    runtime = runtime_record()
    images, ground_truth, predictions = copy.deepcopy((images, ground_truth, predictions))
    ids = [x['id'] for x in images]
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 1 for i in ids):
        raise ValueError('Image IDs must be unique positive integers')
    for row in images:
        if not all(math.isfinite(float(row[k])) and row[k] > 0 for k in ['width', 'height']):
            raise ValueError('Image dimensions must be finite and positive')
    id_set = set(ids)
    annotation_ids = []
    positives, ignored = Counter(), Counter()
    for row in ground_truth:
        annotation_ids.append(row['id'])
        if type(row['id']) is not int or row['id'] < 1:
            raise ValueError('Annotation IDs must be positive integers')
        if row['image_id'] not in id_set or row['category_id'] not in CLASSES:
            raise ValueError('Unknown ground-truth image or category')
        if row.get('iscrowd') not in (0, 1) or 'iscrowd' not in row:
            raise ValueError('An explicit iscrowd=0/1 is required')
        if 'ignore' in row:
            raise ValueError('Ambiguous ignore field: provide an explicit adapter policy')
        area = _bbox(row)
        if not math.isclose(float(row['area']), area, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('Ground-truth area differs from bbox area')
        (ignored if row['iscrowd'] else positives)[row['category_id']] += 1
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError('Duplicate annotation IDs')
    if any(positives[c] == 0 for c in CLASSES):
        raise ValueError('Both classes must have non-ignored ground truth; missing is not zero AP')
    counts = Counter()
    for row in predictions:
        if row['image_id'] not in id_set or row['category_id'] not in CLASSES:
            raise ValueError('Unknown prediction image or category')
        _bbox(row)
        if not math.isfinite(float(row['score'])) or not 0 <= row['score'] <= 1:
            raise ValueError('Prediction confidence must be finite and within [0,1]')
        counts[row['image_id']] += 1
    if any(n > MAX_DET for n in counts.values()):
        raise ValueError('Predictions exceed the frozen all-class inference cap; do not silently truncate')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'info': {}, 'images': images, 'annotations': ground_truth,
                      'categories': [{'id': c, 'name': n} for c, n in CLASSES.items()]}
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': images, 'annotations': [], 'categories': gt.dataset['categories']}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        p = evaluator.params
        p.imgIds = sorted(ids)
        p.catIds = sorted(CLASSES)
        p.iouThrs, p.recThrs = IOU.copy(), RECALL.copy()
        p.maxDets = [MAX_DET]
        p.areaRng, p.areaRngLbl = [[0., 1e10]], ['all']
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval['precision'][:, :, :, 0, 0]
    recall = evaluator.eval['recall'][:, :, 0, 0]
    if precision.shape != (10, 101, 2) or not np.isfinite(precision).all() or np.any(precision < 0):
        raise ValueError('Missing or invalid precision cells')
    result = {'metric': 'Common two-class COCO-style AP101 at IoU .50:.05:.95, maxDet=300, all areas',
              'runtime': runtime, 'images': len(ids), 'predictions': len(predictions),
              'positive_ground_truth': {CLASSES[c]: positives[c] for c in CLASSES},
              'crowd_ignore_ground_truth': {CLASSES[c]: ignored[c] for c in CLASSES},
              'class_order': [CLASSES[c] for c in sorted(CLASSES)],
              'iou_thresholds': IOU.tolist(), 'recall_thresholds': RECALL.tolist(),
              'mean_AP50_95': float(precision.mean()),
              'class_AP50_95': {CLASSES[c]: float(precision[:, :, i].mean()) for i, c in enumerate(sorted(CLASSES))},
              'class_AP_by_IoU': {CLASSES[c]: precision[:, :, i].mean(1).tolist() for i, c in enumerate(sorted(CLASSES))},
              'precision_101': precision.tolist(), 'max_recall_by_IoU': recall.tolist(),
              'annotation_policy_must_be_reported_separately': True}
    if include_matches:
        result['image_matches'] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in entry.items()} for entry in evaluator.evalImgs if entry is not None]
    return result
