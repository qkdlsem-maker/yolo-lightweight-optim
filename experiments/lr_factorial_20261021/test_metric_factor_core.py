"""Synthetic-only known answers and exact parity with the frozen common scorer."""
import argparse,copy,hashlib,importlib.util,json,math,sys
from pathlib import Path
from metric_factor_core import evaluate_grid,evaluate_factorial,runtime_record
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
assert not args.output.exists(),'Never overwrite a validation record'
reference_bytes=args.reference.read_bytes();assert hashlib.sha256(reference_bytes).hexdigest()=='18b1978e94e4fd614feb06cad2d03700114b0ee304a20448c50e1e01bd7c3575'
spec=importlib.util.spec_from_file_location('frozen_common_reference',args.reference);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
ROOT=Path(__file__).resolve().parent
bound={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),ROOT/'metric_factor_core.py',args.reference]}
parity_cases=0
def evaluate(images,gt,dt,include_matches=False):
    global parity_cases
    new=evaluate_grid(images,gt,dt,101,include_matches)
    old=reference.evaluate(images,gt,dt,include_matches)
    assert new['precision_grid']==old['precision_101']
    assert new['max_recall_by_IoU']==old['max_recall_by_IoU']
    assert new['overlap_cells']['iou_grid']['mean_AP']==old['mean_AP50_95']
    assert new['overlap_cells']['iou_grid']['class_AP']==old['class_AP50_95']
    assert new['class_AP_by_IoU']==old['class_AP_by_IoU']
    if include_matches:assert new['image_matches']==old['image_matches']
    parity_cases+=1
    return {**new,'mean_AP50_95':new['overlap_cells']['iou_grid']['mean_AP']}

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
collapsed = [dict(image_id=3, category_id=2, bbox=[5, 5, 0, 10], score=.99)]
check('zero_area_prediction_remains_false_positive', collapsed + DT, .75)
check('zero_area_prediction_is_not_absorbed_by_crowd', collapsed + DT, .75, crowd)
check('zero_area_predictions_cannot_match_ordinary_targets', [dict(DT[0], bbox=[0, 0, 0, 10]), DT[1]], .5)

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
reject('negative_width_prediction', dt=[dict(DT[0], bbox=[0, 0, -1, 10])])
reject('zero_area_ground_truth', gt=[dict(GT[0], bbox=[0, 0, 0, 10], area=0), GT[1]])
reject('ambiguous_ignore', gt=[dict(GT[0], ignore=1), GT[1]])
reject('inference_cap_exceeded', dt=[copy.deepcopy(DT[0]) for _ in range(301)])
reject('missing_image_not_silently_dropped', images=IMAGES[1:])


# Independent known answers for both recall grids and both overlap policies.
def scalar_case(name,images,gt,dt,n,overlap,expected):
    before=copy.deepcopy((images,gt,dt));result=evaluate_grid(images,gt,dt,n,True)
    value=result['overlap_cells'][overlap]['mean_AP']
    assert math.isclose(value,expected,abs_tol=1e-12),(name,value,expected)
    assert (images,gt,dt)==before
    tests.append({'name':name,'expected':expected,'actual':value})
    return result
for n in [101,40]:
 for overlap in ['iou_grid','class_specific_iou']:
    scalar_case(f'perfect_{n}_{overlap}',IMAGES,GT,DT,n,overlap,1.)
    scalar_case(f'empty_{n}_{overlap}',IMAGES,GT,[],n,overlap,0.)
    scalar_case(f'partial_recall_{n}_{overlap}',IMAGES,two_cars,DT,n,overlap,(1+(51/101 if n==101 else 20/40))/2)
    scalar_case(f'empty_image_FP_{n}_{overlap}',IMAGES,GT,[dict(DT[0],image_id=3,score=.99)]+DT,n,overlap,.75)
 boundary_gt=[dict(GT[0],bbox=[0,0,7,10],area=70),dict(GT[1],bbox=[0,0,5,10],area=50)]
 scalar_case(f'exact_class_specific_boundaries_{n}',IMAGES,boundary_gt,DT,n,'class_specific_iou',1.)
 scalar_case(f'full_grid_same_boundary_boxes_{n}',IMAGES,boundary_gt,DT,n,'iou_grid',.3)
 scalar_case(f'below_car_threshold_{n}',IMAGES,[dict(boundary_gt[0],bbox=[0,0,6.99999,10],area=69.9999),boundary_gt[1]],DT,n,'class_specific_iou',.5)
 scalar_case(f'below_person_threshold_{n}',IMAGES,[boundary_gt[0],dict(boundary_gt[1],bbox=[0,0,4.99999,10],area=49.9999)],DT,n,'class_specific_iou',.5)
 score=evaluate_grid(IMAGES,GT,DT,n)
 assert score['recall_thresholds']==([i/40 for i in range(1,41)] if n==40 else reference.RECALL.tolist())
 assert len(score['precision_grid'])==10 and len(score['precision_grid'][0])==n

# Eight-cell integrity, crowd absorption of several detections, and all-image FPs.
inputs=copy.deepcopy((IMAGES,crowd,GT,inside+DT));factorial=evaluate_factorial(*inputs,include_matches=True)
assert inputs==(IMAGES,crowd,GT,inside+DT)
assert len(factorial['cells'])==8 and len({(c['ignore_policy'],c['recall_points'],c['overlap_policy']) for c in factorial['cells']})==8
for cell in factorial['cells']:
 expected=1. if cell['ignore_policy']=='source_ignore' else (1+1/3)/2
 assert math.isclose(cell['mean_AP'],expected,abs_tol=1e-12),(cell,expected)
 tests.append({'name':'factorial_'+cell['grid_reference']+'_'+cell['overlap_policy'],'actual':cell['mean_AP'],'expected':expected})

def fail(name,call):
 try:call()
 except ValueError as error:tests.append({'name':name,'rejected':str(error)})
 else:raise AssertionError('Invalid case accepted: '+name)
for n in [True,100,41,40.,'40',None]:fail('reject_unregistered_recall_grid_'+str(n),lambda n=n:evaluate_grid(IMAGES,GT,DT,n))
fail('ignore_factor_cannot_change_targets',lambda:evaluate_factorial(IMAGES,crowd,[dict(GT[0],bbox=[1,0,10,10]),GT[1]],DT))
fail('no_ignore_view_cannot_contain_crowd',lambda:evaluate_factorial(IMAGES,crowd,crowd,DT))
# A real ordinary target wins over an overlapping crowd under both grids.
for n in [101,40]:
 scalar_case(f'ordinary_priority_{n}',IMAGES,GT+[dict(id=3,image_id=1,category_id=2,bbox=[0,0,100,100],area=10000,iscrowd=1)],DT,n,'class_specific_iou',1.)
for path,digest in bound.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
record={'status':'passed','scope':'Synthetic metric core only; no research checkpoint, real prediction archive or dataset score loaded. Archive integration and exact real cached common-cell reproduction remain required.',
        'cases':tests,'exact_reference_parity_cases':parity_cases,'runtime':runtime_record(),'source_sha256':bound,'real_model_results_used':False,'native_KITTI_equivalence_claimed':False}
with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':'passed','cases':len(tests),'exact_reference_parity_cases':parity_cases}))
