"""Known-answer and fail-closed cases; no real model or external scores used."""
import copy
import hashlib
import json
import math
from pathlib import Path
from common_ap_core import evaluate, runtime_record

ROOT = Path(__file__).resolve().parent
IMAGES = [{'id': i, 'width': 100, 'height': 100} for i in [1, 2, 3]]
GT = [dict(id=1, image_id=1, category_id=2, bbox=[0, 0, 10, 10], area=100, iscrowd=0),
      dict(id=2, image_id=2, category_id=0, bbox=[0, 0, 10, 10], area=100, iscrowd=0)]
DT = [dict(image_id=1, category_id=2, bbox=[0, 0, 10, 10], score=.9),
      dict(image_id=2, category_id=0, bbox=[0, 0, 10, 10], score=.9)]
tests = []


def check(name, predictions, expected, gt=GT):
    original = copy.deepcopy((IMAGES, gt, predictions))
    result = evaluate(IMAGES, gt, predictions, include_matches=True)
    assert (IMAGES, gt, predictions) == original, 'Evaluator mutated caller inputs'
    assert math.isclose(result['mean_AP50_95'], expected, abs_tol=1e-12), (name, result['mean_AP50_95'], expected)
    tests.append({'name': name, 'expected': expected, 'actual': result['mean_AP50_95']})
    return result


check('perfect_detection', DT, 1.)
check('no_detections', [], 0.)
check('missing_person_predictions', DT[:1], .5)
check('empty_image_high_score_false_positive', [dict(image_id=3, category_id=2, bbox=[0, 0, 10, 10], score=.99)] + DT, .75)
check('duplicate_after_matched_target', DT + [dict(image_id=1, category_id=2, bbox=[0, 0, 10, 10], score=.8)], 1.)
check('false_positive_before_target', [dict(image_id=1, category_id=2, bbox=[30, 0, 10, 10], score=.99)] + DT, .75)
partial = copy.deepcopy(DT)
partial[0]['bbox'] = [0, 0, 20, 10]
res = check('exact_IoU_half_boundary', partial, .55)
assert math.isclose(res['class_AP_by_IoU']['car'][0], 1., abs_tol=1e-12)
assert res['class_AP_by_IoU']['car'][1:] == [0.] * 9
wrong = copy.deepcopy(DT)
for row in wrong: row['category_id'] = 2 - row['category_id']
check('classes_are_not_pooled', wrong, 0.)
crowd = GT + [dict(id=3, image_id=3, category_id=2, bbox=[0, 0, 100, 100], area=10000, iscrowd=1)]
inside = [dict(image_id=3, category_id=2, bbox=[5, 5, 5, 5], score=.99),
          dict(image_id=3, category_id=2, bbox=[15, 15, 5, 5], score=.98)]
check('crowd_uses_detection_area_and_accepts_multiple', inside + DT, 1., crowd)
overlap = GT + [dict(id=3, image_id=1, category_id=2, bbox=[0, 0, 100, 100], area=10000, iscrowd=1)]
check('ordinary_target_has_priority_over_crowd', DT, 1., overlap)
two_cars = GT + [dict(id=3, image_id=3, category_id=2, bbox=[0, 0, 10, 10], area=100, iscrowd=0)]
check('partial_recall_101_point_grid', DT, (1 + 51 / 101) / 2, two_cars)

def reject(name, images=IMAGES, gt=GT, dt=DT):
    try:
        evaluate(images, gt, dt)
    except ValueError as error:
        tests.append({'name': name, 'rejected': str(error)})
    else:
        raise AssertionError(name + ' was silently accepted')

reject('missing_class_is_not_zero_AP', gt=GT[:1])
reject('unknown_image', dt=[dict(DT[0], image_id=99)])
reject('unknown_category', dt=[dict(DT[0], category_id=1)])
reject('nonfinite_confidence', dt=[dict(DT[0], score=float('nan'))])
reject('duplicate_annotation', gt=GT + [GT[0]])
reject('zero_area_prediction', dt=[dict(DT[0], bbox=[0, 0, 0, 10])])
reject('ambiguous_ignore', gt=[dict(GT[0], ignore=1), GT[1]])
reject('inference_cap_exceeded', dt=[copy.deepcopy(DT[0]) for _ in range(301)])
reject('missing_image_not_silently_dropped', images=IMAGES[1:])

record = {'status': 'passed', 'scope': 'Synthetic core validation only; dataset converters and frozen real-data inference remain pending',
          'cases': tests, 'runtime': runtime_record(), 'real_model_results_used': False,
          'files_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ['common_ap_core.py', 'test_common_ap_core.py']}}
dest = ROOT / 'common_ap_core_validation.json'
dest.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode('utf-8'))
print(json.dumps({'status': 'passed', 'cases': len(tests), 'record': str(dest)}))
