"""Known-answer common-score integration, solely with fabricated labels/boxes."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import INFERENCE, image_metadata, prediction_record
from score_common_archive import SOURCES, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing integration evidence')
    source = Path(__file__).parent
    hashes = {n: sha256_file(source / n) for n in SOURCES}
    with tempfile.TemporaryDirectory(prefix='synthetic_common_score_') as tmp:
        root = Path(tmp)
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
                   'source_sha256': hashes,
                   'annotation_views': {k: v for k, v in annotation_specs.items() if k.startswith('kitti')}}
        scores_dir = root / 'kitti_scores'
        first = run(request, scores_dir)
        assert first['new_scores_computed']
        scores = {n: json.loads((scores_dir / (n + '.json')).read_bytes()) for n in request['annotation_views']}
        expected = {'kitti_source_ignore': 1., 'kitti_no_ignore_sensitivity': .5}
        for name, item in scores.items():
            assert abs(item['score']['mean_AP50_95'] - expected[name]) < 1e-12
            assert item['all_class_predictions_preserved'] == 6
            assert item['mapped_person_car_predictions'] == 5
            assert item['score']['images'] == 2
        assert len({v['shared_common_prediction_projection_sha256'] for v in scores.values()}) == 1
        before = {p.name: p.read_bytes() for p in scores_dir.iterdir()}
        with patch('score_common_archive.evaluate', side_effect=AssertionError('Completed scores must not be recomputed')):
            assert not run(request, scores_dir)['new_scores_computed']
        assert all((scores_dir / n).read_bytes() == raw for n, raw in before.items())
        bdd_request = copy.deepcopy(request)
        bdd_request.update(dataset='bdd', annotation_views={'bdd_common': annotation_specs['bdd_common']})
        run(bdd_request, root / 'bdd_scores')
        bdd_score = json.loads((root / 'bdd_scores/bdd_common.json').read_bytes())
        assert abs(bdd_score['score']['mean_AP50_95'] - .5) < 1e-12
        rejected = []
        missing = copy.deepcopy(request)
        missing['annotation_views'].pop('kitti_no_ignore_sensitivity')
        changed_source = copy.deepcopy(request)
        changed_source['source_sha256']['common_ap_core.py'] = '0' * 64
        for name, bad in [('missing_fixed_variant', missing), ('changed_source', changed_source)]:
            try:
                run(bad, root / ('bad_' + name))
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Expected rejection: ' + name)
        changed = root / 'kitti_no_ignore_sensitivity.json'
        changed.write_bytes(changed.read_bytes() + b' ')
        try:
            run(request, scores_dir)
        except ValueError:
            rejected.append('changed_annotation_even_with_complete_scores')
        else:
            raise AssertionError('Changed annotations accepted')
        changed.write_bytes(changed.read_bytes()[:-1])
        score_file = scores_dir / 'kitti_source_ignore.json'
        score_file.write_bytes(score_file.read_bytes() + b' ')
        try:
            run(request, scores_dir)
        except ValueError:
            rejected.append('changed_score_artifact')
        else:
            raise AssertionError('Changed score output accepted')
        result = {'status': 'passed', 'scope': 'Fabricated image metadata, labels and predictions only; no model or real dataset accessed',
                  'known_answer_AP50_95': {**expected, 'bdd_common': .5},
                  'same_prediction_projection_for_both_KITTI_variants': True,
                  'empty_image_and_zero_area_prediction_retained': True,
                  'all_class_predictions': 6, 'mapped_predictions': 5,
                  'completed_scores_reused_without_recomputation_or_changes': True,
                  'invalid_requests_and_mutations_rejected': rejected,
                  'source_sha256': {**hashes, Path(__file__).name: sha256_file(Path(__file__))},
                  'runtime': bdd_score['score']['runtime'],
                  'real_model_or_dataset_scores_observed': False,
                  'native_BDD_or_native_KITTI_metrics_covered': False}
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
