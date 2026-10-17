"""Pure label conversion under common_annotation_policy.json; no input writes.

Return rows without image/id fields; the dataset builder assigns stable IDs.
Counters record all input rows, including classes excluded from the diagnostic.
"""
import math
from collections import Counter

TARGETS = {0, 2}
KITTI_CATEGORIES = {'Car', 'Pedestrian', 'Van', 'Person_sitting', 'DontCare',
                    'Truck', 'Cyclist', 'Tram', 'Misc'}


def _row(category, x, y, width, height, crowd):
    values = [float(v) for v in [x, y, width, height]]
    if not all(math.isfinite(v) for v in values) or width <= 0 or height <= 0:
        raise ValueError('Nonfinite or nonpositive annotation box')
    return {'category_id': category, 'bbox': values,
            'area': float(width * height), 'iscrowd': int(crowd)}


def bdd_rows(text, image_width, image_height):
    if not all(math.isfinite(float(v)) and v > 0 for v in [image_width, image_height]):
        raise ValueError('Invalid image dimensions')
    rows, counts = [], Counter()
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError('BDD YOLO row requires five fields')
        category = int(fields[0])
        if category not in range(10):
            raise ValueError('Unknown BDD category')
        cx, cy, w, h = map(float, fields[1:])
        box = _row(category, (cx - w / 2) * image_width, (cy - h / 2) * image_height,
                   w * image_width, h * image_height, 0)
        counts[str(category)] += 1
        if category in TARGETS:
            rows.append(box)
    return rows, dict(counts)


def kitti_rows(text, variant='source_ignore_regions'):
    if variant not in ['source_ignore_regions', 'no_ignore_sensitivity']:
        raise ValueError('Unknown prespecified annotation variant')
    rows, counts = [], Counter()
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 15:
            raise ValueError('KITTI ground-truth row requires exactly 15 fields')
        label = fields[0]
        if label not in KITTI_CATEGORIES:
            raise ValueError('Unknown KITTI category: ' + label)
        numeric = list(map(float, fields[1:]))
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('Nonfinite KITTI annotation')
        x1, y1, x2, y2 = map(float, fields[4:8])
        # Validate every source row, even when its class is not evaluated.
        box = _row(0, x1, y1, x2 - x1, y2 - y1, 0)
        counts[label] += 1
        if label in ['Car', 'Pedestrian']:
            box['category_id'] = 2 if label == 'Car' else 0
            rows.append(box)
        elif variant == 'source_ignore_regions':
            categories = {'Van': [2], 'Person_sitting': [0], 'DontCare': [0, 2]}.get(label, [])
            for category in categories:
                rows.append(dict(box, category_id=category, iscrowd=1))
    return rows, dict(counts)


def identify_rows(rows, image_id, first_annotation_id):
    if type(image_id) is not int or image_id < 1 or type(first_annotation_id) is not int or first_annotation_id < 1:
        raise ValueError('IDs must be positive integers')
    return [dict(row, id=first_annotation_id + i, image_id=image_id) for i, row in enumerate(rows)]
