"""Known source-format examples and integration with the common AP core."""
import hashlib
import json
from pathlib import Path
from common_annotation_adapters import bdd_rows, kitti_rows, identify_rows
from common_ap_core import evaluate

ROOT = Path(__file__).resolve().parent
checks = []


def kline(label, xyxy=(10, 20, 30, 60), occlusion=0, truncation=0):
    return ' '.join(map(str, [label, truncation, occlusion, 0, *xyxy, 1, 1, 1, 0, 0, 1, 0]))


rows, counts = bdd_rows('2 .5 .5 .25 .5\n0 .25 .25 .125 .25\n1 .5 .5 .1 .1\n', 1280, 720)
assert rows[0]['bbox'] == [480., 180., 320., 360.]
assert rows[1]['bbox'] == [240., 90., 160., 180.]
assert len(rows) == 2 and counts == {'2': 1, '0': 1, '1': 1}
checks.append('BDD pixel coordinates and rider exclusion')
duplicates, _ = bdd_rows('2 .5 .5 .2 .2\n2 .5 .5 .2 .2', 100, 100)
assert duplicates[0] == duplicates[1] and len(duplicates) == 2
checks.append('BDD duplicate multiplicity preserved')
outside, _ = bdd_rows('2 0 0 .2 .2', 100, 100)
assert outside[0]['bbox'] == [-10., -10., 20., 20.]
checks.append('Source boxes are not silently clipped')
text = '\n'.join(kline(c) for c in ['Car', 'Pedestrian', 'Van', 'Person_sitting', 'DontCare', 'Truck', 'Cyclist', 'Tram', 'Misc'])
source_ignore, count_a = kitti_rows(text)
strict, count_b = kitti_rows(text, 'no_ignore_sensitivity')
assert count_a == count_b and sum(count_a.values()) == 9
assert [(r['category_id'], r['iscrowd']) for r in source_ignore] == [(2, 0), (0, 0), (2, 1), (0, 1), (0, 1), (2, 1)]
assert [r for r in source_ignore if not r['iscrowd']] == strict and len(strict) == 2
assert all(r['bbox'] == [10., 20., 20., 40.] and r['area'] == 800. for r in source_ignore)
checks.append('KITTI exact category and two-variant mapping, no inclusive-pixel offset')
difficult, _ = kitti_rows(kline('Car', (0, 0, 10, 5), occlusion=3, truncation=.99))
assert len(difficult) == 1 and difficult[0]['iscrowd'] == 0
checks.append('Common metric does not apply native difficulty filters')
assert identify_rows(strict, 3, 12)[1]['id'] == 13
checks.append('Stable explicit image and annotation IDs')

images = [{'id': 1, 'width': 100, 'height': 100}]
synthetic = '\n'.join([kline('Car', (0, 0, 10, 10)), kline('Pedestrian', (20, 0, 30, 10)),
                       kline('Van', (40, 0, 60, 20)), kline('DontCare', (60, 60, 100, 100))])
predictions = [dict(image_id=1, category_id=c, bbox=box, score=s) for c, box, s in [
    (2, [45, 5, 5, 5], .99), (0, [70, 70, 5, 5], .98),
    (2, [0, 0, 10, 10], .9), (0, [20, 0, 10, 10], .9)]]
scores = {}
for variant in ['source_ignore_regions', 'no_ignore_sensitivity']:
    converted, _ = kitti_rows(synthetic, variant)
    scores[variant] = evaluate(images, identify_rows(converted, 1, 1), predictions)['mean_AP50_95']
assert abs(scores['source_ignore_regions'] - 1.) < 1e-12
assert abs(scores['no_ignore_sensitivity'] - .5) < 1e-12
checks.append('Both fixed ignore variants produce their known different AP values')

bad = [lambda: bdd_rows('2 .5 .5 0 .1', 100, 100),
       lambda: bdd_rows('2 .5 .5 .1 nan', 100, 100),
       lambda: bdd_rows('10 .5 .5 .1 .1', 100, 100),
       lambda: kitti_rows(kline('Unknown')),
       lambda: kitti_rows(kline('Car') + ' .9'),
       lambda: kitti_rows(kline('Car', (10, 0, 10, 10))),
       lambda: kitti_rows(kline('Car'), 'posthoc'),
       lambda: identify_rows([], 0, 1)]
for i, f in enumerate(bad):
    try:
        f()
    except ValueError:
        checks.append(f'Malformed-input rejection {i + 1}')
    else:
        raise AssertionError(f'Malformed input {i + 1} accepted')
record = {'status': 'passed', 'scope': 'Synthetic label conversion and AP integration only; full source-label build remains pending',
          'checks': checks, 'synthetic_variant_scores': scores, 'real_model_results_used': False,
          'files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                           ['common_annotation_policy.json', 'common_annotation_adapters.py', 'test_common_annotation_adapters.py', 'common_ap_core.py']}}
(ROOT / 'common_annotation_validation.json').write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
print(json.dumps({'status': 'passed', 'checks': len(checks)}))
