"""Synthetic integration only; reuses the already verified original 2D binary."""
import argparse
import copy
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import INFERENCE, prediction_record
from score_native_kitti_archive import CORE, SOURCES, inventory, parse_curves, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--core-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing integration proof')
    source = Path(__file__).parent
    core = args.core_root.resolve()
    core_paths = {
        'executable': str(core / 'local_evaluator/evaluate_2d_local'),
        'original_source': str(core / 'local_evaluator/evaluate_object_original.cpp'),
        'adapted_source': str(core / 'local_evaluator/evaluate_2d_local.cpp'),
        'build_manifest': str(core / 'local_evaluator/build_manifest.json'),
        'prior_synthetic_proof': str(core / 'evaluator_verification.json'),
    }
    hashes = {name: sha256_file(source / name) for name in SOURCES}
    assert {k: sha256_file(v) for k, v in core_paths.items()} == CORE
    with tempfile.TemporaryDirectory(prefix='synthetic_native_kitti_archive_') as temp:
        root = Path(temp)
        images = [{'id': i + 1, 'file_name': f'{i:06d}.png', 'width': 600, 'height': 100,
                   'sha256': digest(str(i).encode())} for i in range(400)]
        bindings = {'endpoint_id': 'synthetic_only', 'dataset_id': 'fabricated_only',
                    'checkpoint_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64,
                    'source_sha256': {'synthetic_only': 'c' * 64},
                    'runtime': {'purpose': 'known_answer_test_no_model'}, 'inference': INFERENCE}
        rows = [[0, 0, 60, 60, .9, 2], [100, 0, 160, 60, .9, 0],
                [200, 0, 260, 60, .99, 2], [200, 0, 260, 60, .99, 0],
                [300, 0, 360, 60, .999, 9], [0, 0, 0, 0, .1, 2], [0, 0, 0, 0, .1, 0]]
        archive_path = root / 'archive'
        with PredictionArchive(archive_path, bindings, images) as archive:
            for start in range(0, 400, 32):
                batch = images[start:start + 32]
                records = [prediction_record(im, rows if (im['id'] - 1) % 2 == 0 else []) for im in batch]
                archive.commit(start // 32, records,
                               [{'shape': [len(batch), 3, 640, 640], 'dtype': 'torch.float32', 'device': 'cpu'}])
            archive.finalize()
        label = (b'Car 0 0 -10 0 0 60 60 -1 -1 -1 -1000 -1000 -1000 -10\n'
                 b'Pedestrian 0 0 -10 100 0 160 60 -1 -1 -1 -1000 -1000 -1000 -10\n'
                 b'DontCare 0 0 -10 200 0 260 60 -1 -1 -1 -1000 -1000 -1000 -10\n')
        labels_zip = root / 'fabricated_labels.zip'
        with zipfile.ZipFile(labels_zip, 'w') as z:
            for i in range(400):
                z.writestr(f'training/label_2/{i:06d}.txt', label)
        request = {'schema': 'native_kitti_archive_scoring_request_v1',
                   'archive_path': str(archive_path),
                   'archive_manifest_sha256': sha256_file(archive_path / 'MANIFEST.json'),
                   'archive_complete_sha256': sha256_file(archive_path / 'COMPLETE.json'),
                   'source_sha256': hashes, 'native_core_paths': core_paths,
                   'label_zip': {'path': str(labels_zip), 'sha256': sha256_file(labels_zip)}}
        output = root / 'native_scores'
        # A planned process failure must leave an inspectable partial directory,
        # never a completed score. The successful retry uses a new stage.
        with patch('score_native_kitti_archive.subprocess.run', side_effect=subprocess.CalledProcessError(7, ['synthetic_failure'])):
            try:
                run(request, output)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('Injected failure was not surfaced')
        assert not output.exists()
        partials = list(root.glob('native_scores.partial-*'))
        assert len(partials) == 1 and not (partials[0] / 'COMPLETE.json').exists()
        partial_hashes = inventory(partials[0])
        result = run(request, output)
        assert result['new_scores_computed']
        score = json.loads((output / 'score.json').read_bytes())
        assert abs(score['mean_moderate_AP_R40'] - .5) < 1e-12
        for cls in ('car', 'pedestrian'):
            for diff in ('easy', 'moderate', 'hard'):
                assert abs(score['classes'][cls][diff]['AP_R40'] - .5) < 1e-12
        assert score['images'] == 400 and score['all_class_predictions_preserved'] == 1400
        assert score['mapped_predictions_exported'] == 1200
        assert score['empty_prediction_files'] == 200 and score['zero_area_mapped_predictions_exported'] == 400
        assert len(list((output / 'predictions').glob('*.txt'))) == 400
        assert all((output / 'ground_truth' / f'{i:06d}.txt').read_bytes() == label for i in range(400))
        assert all((output / 'predictions' / f'{i:06d}.txt').read_bytes() == b'' for i in range(1, 400, 2))
        assert inventory(partials[0]) == partial_hashes
        before = inventory(output)
        completion = (output / 'COMPLETE.json').read_bytes()
        with patch('score_native_kitti_archive.subprocess.run', side_effect=AssertionError('Complete scores must not execute binary again')):
            assert not run(request, output)['new_scores_computed']
        assert before == inventory(output) and completion == (output / 'COMPLETE.json').read_bytes()
        rejected = []
        bad = copy.deepcopy(request)
        bad['source_sha256']['score_native_kitti_archive.py'] = '0' * 64
        try:
            run(bad, output)
        except ValueError:
            rejected.append('changed_source')
        else:
            raise AssertionError('Changed source accepted')
        raw_zip = labels_zip.read_bytes()
        labels_zip.write_bytes(raw_zip + b' ')
        try:
            run(request, output)
        except ValueError:
            rejected.append('changed_labels_even_with_completed_scores')
        else:
            raise AssertionError('Changed labels accepted')
        labels_zip.write_bytes(raw_zip)
        altered_text = output / 'predictions/000000.txt'
        altered_text.write_bytes(altered_text.read_bytes() + b' ')
        try:
            run(request, output)
        except ValueError:
            rejected.append('changed_exported_prediction_text')
        else:
            raise AssertionError('Changed native text accepted')
        curves = root / 'invalid_curves'
        curves.mkdir()
        for cls in ('car', 'pedestrian'):
            (curves / f'stats_{cls}_detection.txt').write_text(('0 ' * 41 + '\n') * 3)
        for case, data in [('wrong_precision_shape', '0\n'), ('nonfinite_precision', ('nan ' * 41 + '\n') * 3)]:
            (curves / 'stats_car_detection.txt').write_text(data)
            try:
                parse_curves(curves)
            except ValueError:
                rejected.append(case)
            else:
                raise AssertionError('Invalid curves accepted')
        proof = {'status': 'passed', 'scope': 'Fabricated image metadata, labels and predictions only; no model or real dataset evaluated',
                 'native_core_sha256': CORE, 'existing_eight_case_proof_reused_without_reexecution': True,
                 'images': 400, 'known_answer_AP_R40_all_classes_and_difficulties': .5,
                 'all_class_predictions': 1400, 'mapped_predictions': 1200,
                 'empty_prediction_files': 200, 'zero_area_mapped_predictions_preserved': 400,
                 'GT_ZIP_member_bytes_preserved': True,
                 'injected_process_failure_preserved_as_partial_without_complete': True,
                 'completed_outputs_reused_without_execution_or_mutation': True,
                 'invalid_requests_and_mutations_rejected': rejected,
                 'source_sha256': {**hashes, Path(__file__).name: sha256_file(Path(__file__))},
                 'real_model_or_dataset_scores_observed': False}
    args.output.write_bytes((json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
