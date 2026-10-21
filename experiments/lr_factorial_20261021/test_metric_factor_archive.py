"""Archive integration with fabricated predictions only; fixtures are preserved."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
from prediction_archive import PredictionArchive,canonical,digest,sha256_file
from prediction_io import INFERENCE,image_metadata,prediction_record
import score_metric_factor_archive as scorer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists() and not args.fixtures.exists()
    root=args.fixtures;root.mkdir()
    sys.path.append(str(args.reference))
    spec=importlib.util.spec_from_file_location('prior_common_score',args.reference/'score_common_archive.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    source=Path(__file__).parent;hashes={n:sha256_file(source/n) for n in scorer.SOURCES}
    common_hashes={n:sha256_file(args.reference/n) for n in prior.SOURCES}
    images = [{'id': i, 'file_name': f'{i - 1:06d}.png', 'width': 300, 'height': 100,
               'sha256': digest(str(i).encode())} for i in (1, 2)]
    binding = {'endpoint_id': 'synthetic_only', 'dataset_id': 'fabricated_only',
               'checkpoint_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64,
               'source_sha256': {'synthetic_only': 'c' * 64},
               'runtime': {'purpose': 'known_answer_test_no_model'}, 'inference': INFERENCE}
    archive_path = root / 'predictions'
    rows = [[0, 0, 20, 40, .8, 0], [50, 0, 70, 40, .8, 2],
            [100, 0, 120, 40, .9, 0], [150, 0, 170, 40, .9, 2],
            [200, 0, 220, 40, .99, 9], [0, 0, 0, 0, .1, 0]]
    with PredictionArchive(archive_path, binding, images) as archive:
        archive.commit(0, [prediction_record(images[0], rows), prediction_record(images[1], [])],
                       [{'shape': [2, 3, 640, 640], 'dtype': 'torch.float32', 'device': 'cpu'}])
        archive.finalize()
    def gt(index, category, x, ignored=0):
        return {'id': index, 'image_id': 1, 'category_id': category,
                'bbox': [x, 0, 20, 40], 'area': 800, 'iscrowd': ignored}
    positive = [gt(1, 0, 0), gt(2, 2, 50)]
    variants = {'kitti_source_ignore': positive + [gt(3, 0, 100, 1), gt(4, 2, 150, 1)],
                'kitti_no_ignore_sensitivity': positive,
                'bdd_common': positive}
    annotation_specs = {}
    for name, labels in variants.items():
        data = {'images': [image_metadata(x) for x in images], 'annotations': labels,
                'categories': [{'id': 0, 'name': 'person'}, {'id': 2, 'name': 'car'}],
                'info': {'annotation_variant': name, 'policy_sha256': hashes['common_annotation_policy.json']}}
        path = root / (name + '.json')
        path.write_bytes(canonical(data))
        annotation_specs[name] = {'path': str(path), 'sha256': sha256_file(path)}
    request = {'schema': 'common_archive_scoring_request_v1', 'dataset': 'kitti',
               'archive_path': str(archive_path), 'archive_manifest_sha256': sha256_file(archive_path / 'MANIFEST.json'),
               'archive_complete_sha256': sha256_file(archive_path / 'COMPLETE.json'),
               'source_sha256': common_hashes,
               'annotation_views': {k: v for k, v in annotation_specs.items() if k.startswith('kitti')}}
    prior_output=root/'prior_common_scores';prior.run(request,prior_output)
    reference_specs={v:{'path':str(prior_output/(v+'.json')),'sha256':sha256_file(prior_output/(v+'.json'))} for v in scorer.VARIANTS}
    request.update(schema='metric_factor_archive_request_v1',source_sha256=hashes,reference_common_scores=reference_specs)
    output=root/'factorial_scores'
    before_inputs={str(x):x.read_bytes() for x in root.rglob('*') if x.is_file() and x.name!='.lock'}
    result=scorer.run(request,output);assert result['new_scores_computed'] and result['cells']==8
    artifact=json.loads((output/'factorial.json').read_bytes())
    assert len(artifact['cells'])==8 and len(artifact['grids'])==4
    for cell in artifact['cells']:
        expected=1. if cell['ignore_policy']=='source_ignore' else .5
        assert abs(cell['mean_AP']-expected)<1e-12
    assert artifact['all_class_predictions_preserved']==6 and artifact['mapped_person_car_predictions']==5
    assert all(g['images']==2 for g in artifact['grids'].values())
    assert artifact['common_reference_cells_exactly_reproduced']==list(scorer.VARIANTS)
    assert all(Path(path).read_bytes()==raw for path,raw in before_inputs.items())
    before={x.name:x.read_bytes() for x in output.iterdir()}
    with patch('score_metric_factor_archive.evaluate_grid',side_effect=AssertionError('Completed scores recomputed')):
        assert not scorer.run(request,output)['new_scores_computed']
    assert all((output/n).read_bytes()==raw for n,raw in before.items())
    rejected=[]
    def reject(name,bad,out=None):
        try:scorer.run(bad,out or root/('bad_'+name))
        except (ValueError,FileNotFoundError):rejected.append(name)
        else:raise AssertionError('Invalid request accepted: '+name)
    bad=copy.deepcopy(request);bad['annotation_views'].pop('kitti_no_ignore_sensitivity');reject('missing_view',bad)
    bad=copy.deepcopy(request);bad['reference_common_scores'].pop('kitti_source_ignore');reject('missing_common_reference',bad)
    bad=copy.deepcopy(request);bad['source_sha256']['metric_factor_core.py']='0'*64;reject('source_changed',bad)
    bad=copy.deepcopy(request);bad['chosen_overlap']='class_specific_iou';reject('extra_selection_option',bad)
    ref=json.loads((prior_output/'kitti_source_ignore.json').read_bytes());ref['score']['mean_AP50_95']=.123
    altered=root/'synthetic_wrong_common_score.json';altered.write_bytes(canonical(ref))
    bad=copy.deepcopy(request);bad['reference_common_scores']['kitti_source_ignore']={'path':str(altered),'sha256':sha256_file(altered)}
    with patch('score_metric_factor_archive.evaluate_grid',wraps=scorer.evaluate_grid) as observed:
        reject('numerically_wrong_common_reference',bad)
        assert len(observed.call_args_list)==1 and observed.call_args_list[0].args[3]==101,'Must reject before any 40-point rescoring'
    ref['shared_common_prediction_projection_sha256']='0'*64;altered2=root/'wrong_projection.json';altered2.write_bytes(canonical(ref))
    bad=copy.deepcopy(request);bad['reference_common_scores']['kitti_source_ignore']={'path':str(altered2),'sha256':sha256_file(altered2)}
    with patch('score_metric_factor_archive.evaluate_grid',side_effect=AssertionError('Wrong projection reached evaluator')):reject('different_prediction_projection',bad)
    file=prior_output/'kitti_source_ignore.json';raw=file.read_bytes();file.write_bytes(raw+b' ')
    reject('changed_common_reference_after_completion',request,output);file.write_bytes(raw)
    file=Path(request['annotation_views']['kitti_source_ignore']['path']);raw=file.read_bytes();file.write_bytes(raw+b' ')
    reject('changed_annotations_after_completion',request,output);file.write_bytes(raw)
    file=output/'factorial.json';file.write_bytes(file.read_bytes()+b' ')
    reject('changed_score_artifact',request,output)
    result={'status':'passed','scope':'Fabricated labels/image metadata/all-class prediction archive only; no real research predictions or images loaded.',
            'known_answer_all_eight_cells':True,'common_reference_exact_reproduction':True,'wrong_common_reference_rejected_before_40_point_rescoring':True,
            'images_including_empty':2,'all_class_predictions':6,'mapped_predictions_including_zero_area':5,
            'completed_output_reused_without_recomputation':True,'inputs_unchanged_by_scoring':True,'rejected':rejected,
            'source_sha256':{**hashes,Path(__file__).name:sha256_file(Path(__file__))},'original_reference_sources':common_hashes,
            'runtime':scorer.runtime_record(),'real_model_or_dataset_scores_observed':False,'native_KITTI_equivalence_claimed':False,
            'fixtures_preserved':str(root),'intentional_corruption_only_within_synthetic_fixture':True}
    with args.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps({'status':'passed','rejected':len(rejected),'cells':8,'fixtures':str(root)}))

if __name__=='__main__':main()
