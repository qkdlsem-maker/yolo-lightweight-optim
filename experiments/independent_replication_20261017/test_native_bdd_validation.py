"""CPU integration on generated images/labels, including empty/duplicate rows."""
import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from prediction_archive import canonical, sha256_file
from prediction_io import INFERENCE
from run_native_bdd_validation import SOURCES, inventory, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior proof')
    source = Path(__file__).parent
    checkpoint_sha = sha256_file(a.checkpoint)
    names = YOLO(str(a.checkpoint)).names
    with tempfile.TemporaryDirectory(prefix='synthetic_native_bdd_') as tmp:
        root = Path(tmp)
        images_dir, labels_dir = root / 'images', root / 'labels'
        images_dir.mkdir()
        labels_dir.mkdir()
        images = []
        h = hashlib.sha256()
        for i in range(33):
            name = f'synthetic_{i:03d}'
            image_path = images_dir / (name + '.png')
            Image.fromarray(np.full((720, 1280, 3), (i % 3) * 127, np.uint8)).save(image_path)
            images.append({'id': i + 1, 'file_name': image_path.name, 'width': 1280, 'height': 720,
                           'sha256': sha256_file(image_path)})
            row = f'{i % 10} 0.5 0.5 0.1 0.1\n'.encode()
            raw = b'' if i == 0 else row * (2 if i == 1 else 1)
            (labels_dir / (name + '.txt')).write_bytes(raw)
            h.update((name + '.txt').encode() + b'\0' + raw)
        image_manifest = root / 'image_manifest.json'
        image_manifest.write_bytes(canonical(images))
        dataset_manifest = root / 'dataset_manifest.json'
        dataset_manifest.write_bytes(canonical({'splits': {'val': {'images': 33, 'boxes': 33,
                                                                  'ordered_label_content_sha256': h.hexdigest()}}}))
        data_yaml = root / 'fixture.yaml'
        data_yaml.write_bytes(yaml.safe_dump({'names': names, 'fixture_only': True}).encode())
        protocol = root / 'fixture_protocol.json'
        protocol.write_bytes(canonical({'scope': 'Synthetic integration fixture only; not research protocol',
                                        'evaluation': {'inference': INFERENCE},
                                        'data': {'yaml_sha256': sha256_file(data_yaml),
                                                 'manifest_sha256': sha256_file(dataset_manifest), 'validation_images': 33}}))
        spec = {'schema': 'native_bdd_validation_request_v1', 'endpoint_id': 'synthetic_smoke_only',
                'image_dir': str(images_dir), 'label_dir': str(labels_dir),
                'source_sha256': {n: sha256_file(source / n) for n in SOURCES}}
        for key, path in [('checkpoint', a.checkpoint), ('protocol', protocol), ('image_manifest', image_manifest),
                          ('dataset_manifest', dataset_manifest), ('data_yaml', data_yaml)]:
            spec[key] = {'path': str(path), 'sha256': sha256_file(path)}
        original_labels = {p.name: p.read_bytes() for p in labels_dir.glob('*.txt')}
        output = root / 'native_validation'
        assert run(spec, output, 'cpu')['new_inference']
        audit = json.loads((output / 'input_audit.json').read_bytes())
        score = json.loads((output / 'score.json').read_bytes())
        assert audit['images'] == list(range(1, 34))
        assert audit['native_loaded_labels_match_raw_float32_unique_rows']
        assert audit['raw_label_rows'] == 33 and audit['native_loaded_boxes'] == 32
        assert not audit['training_context'] and audit['workers'] == 0
        assert [x['shape'] for x in audit['actual_forward_inputs']] == [[32, 3, 640, 640], [1, 3, 640, 640]]
        assert all(x['dtype'] == 'torch.float32' and x['device'] == 'cpu' for x in audit['actual_forward_inputs'])
        assert [len(x['images']) for x in audit['batches']] == [32, 1]
        assert audit['batches'][0]['images'][0]['targets'] == 0
        assert audit['batches'][0]['images'][1]['targets'] == 1
        saved = [np.load(p, allow_pickle=False) for p in sorted((output / 'native_stats').glob('*.npz'))]
        arrays = {key: np.concatenate([record[key] for record in saved]) for key in saved[0].files}
        for record in saved:
            record.close()
        assert len(arrays['target_image_id']) == len(arrays['target_cls']) == 32
        assert len(arrays['prediction_image_id']) == len(arrays['conf']) == len(arrays['tp'])
        replay = DetMetrics(names=names)
        if arrays['tp'].any():
            replay.process(**{k: arrays[k] for k in ('tp', 'conf', 'pred_cls', 'target_cls')})
        assert replay.results_dict == score['results'], (replay.results_dict, score['results'])
        assert not (labels_dir.with_suffix('.cache')).exists()
        assert not list(labels_dir.glob('*.cache'))
        assert {p.name: p.read_bytes() for p in labels_dir.glob('*.txt')} == original_labels
        assert sha256_file(a.checkpoint) == checkpoint_sha
        before, completion = inventory(output), (output / 'COMPLETE.json').read_bytes()
        with patch('ultralytics.YOLO', side_effect=AssertionError('Complete validation must not reload model')):
            assert not run(spec, output, 'cpu')['new_inference']
        assert inventory(output) == before and (output / 'COMPLETE.json').read_bytes() == completion
        rejected = []
        bad = copy.deepcopy(spec)
        bad['source_sha256']['run_native_bdd_validation.py'] = '0' * 64
        try:
            run(bad, output, 'cpu')
        except ValueError:
            rejected.append('changed_source')
        else:
            raise AssertionError('Source mismatch accepted')
        label = labels_dir / 'synthetic_001.txt'
        label.write_bytes(label.read_bytes() + b'\n')
        try:
            run(spec, output, 'cpu')
        except ValueError:
            rejected.append('changed_original_labels_even_with_complete_output')
        else:
            raise AssertionError('Label mutation accepted')
        label.write_bytes(original_labels[label.name])
        image = images_dir / images[0]['file_name']
        image_raw = image.read_bytes()
        image.write_bytes(image_raw + b' ')
        try:
            run(spec, output, 'cpu')
        except ValueError:
            rejected.append('changed_original_image_even_with_complete_output')
        else:
            raise AssertionError('Image mutation accepted')
        image.write_bytes(image_raw)
        stats = output / 'native_stats/batch_0000.npz'
        stats.write_bytes(stats.read_bytes() + b' ')
        try:
            run(spec, output, 'cpu')
        except ValueError:
            rejected.append('changed_native_statistics_artifact')
        else:
            raise AssertionError('Output mutation accepted')
        manifest = json.loads((output / 'MANIFEST.json').read_bytes())
        proof = {'status': 'passed', 'scope': 'CPU standalone native validation of generated33image fixtures only',
                 'real_BDD_KITTI_data_used': False, 'synthetic_accuracy_or_latency_reported_as_research': False,
                 'images': 33, 'raw_label_rows': 33, 'native_unique_label_rows': 32,
                 'empty_image_retained': True, 'native_duplicate_label_handling_verified': True,
                 'native_loaded_labels_match_pinned_float32_rows': True,
                 'actual_forward_inputs': audit['actual_forward_inputs'], 'setup_probes': audit['setup_probes'],
                 'preserved_native_statistics_reproduce_native_results_exactly': True,
                 'completed_output_reused_without_loading_model_or_inference': True,
                 'original_labels_and_checkpoint_unchanged': True, 'original_label_cache_not_created': True,
                 'invalid_requests_and_mutations_rejected': rejected,
                 'preflight_checkpoint_sha256': checkpoint_sha,
                 'native_validation_source_sha256': manifest['native_validation_source_sha256'],
                 'runtime': manifest['runtime'],
                 'source_sha256': {**spec['source_sha256'], Path(__file__).name: sha256_file(Path(__file__))}}
    a.output.write_bytes((json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
