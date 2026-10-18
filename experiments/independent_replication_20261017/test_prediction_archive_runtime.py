"""CPU synthetic inference: compare resumed and uninterrupted archive bytes."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from prediction_archive import canonical, sha256_file
from run_prediction_archive import SOURCES, run


class SyntheticInterruption(RuntimeError):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous integration evidence')
    source = Path(__file__).parent
    initial_checkpoint_hash = sha256_file(a.checkpoint)
    with tempfile.TemporaryDirectory(prefix='synthetic_archive_integration_') as tmp:
        root = Path(tmp)
        image_dir = root / 'images'
        image_dir.mkdir()
        images = []
        for i in range(33):
            width, height = [(1280, 720), (1242, 375), (1241, 376)][i % 3]
            name = f'synthetic_{i:03d}.png'
            path = image_dir / name
            Image.fromarray(np.full((height, width, 3), (i % 3) * 127, np.uint8)).save(path)
            images.append({'id': i + 1, 'file_name': name, 'width': width, 'height': height,
                           'sha256': sha256_file(path)})
        manifest = root / 'images.json'
        manifest.write_bytes(canonical(images))
        spec = {'schema': 'frozen_prediction_request_v1', 'endpoint_id': 'synthetic_smoke_only',
                'dataset_id': 'generated_unlabeled_images_only', 'image_dir': str(image_dir),
                'checkpoint': {'path': str(a.checkpoint), 'sha256': initial_checkpoint_hash},
                'protocol': {'path': str(a.protocol), 'sha256': sha256_file(a.protocol)},
                'image_manifest': {'path': str(manifest), 'sha256': sha256_file(manifest)},
                'source_sha256': {n: sha256_file(source / n) for n in SOURCES}}
        resumed = root / 'resumed'
        def interrupt(index):
            if index == 0:
                raise SyntheticInterruption('Deliberate synthetic interruption after durable batch0 commit')
        try:
            run(spec, resumed, 'cpu', after_commit=interrupt)
        except SyntheticInterruption:
            pass
        else:
            raise AssertionError('Expected the synthetic interruption')
        assert not (resumed / 'COMPLETE.json').exists()
        saved = {str(p.relative_to(resumed)): p.read_bytes() for p in (resumed / 'batches').rglob('*.json')}
        assert len(saved) == 33  #32 predictions and one batch commit.
        recovery = run(spec, resumed, 'cpu')
        assert recovery['status'] == 'complete'
        assert recovery['previous_committed_batches'] == 1
        assert recovery['new_batch_indices'] == [1] and recovery['new_image_ids'] == [33]
        assert [t['shape'] for t in recovery['actual_forward_inputs']] == [[1, 3, 640, 640]]
        assert all((resumed / name).read_bytes() == raw for name, raw in saved.items())
        untouched = run(spec, resumed, 'cpu')
        assert untouched['archive_complete'] and not untouched['checkpoint_loaded']
        assert untouched['new_batch_indices'] == [] and untouched['actual_forward_inputs'] == []
        straight = root / 'uninterrupted'
        normal = run(spec, straight, 'cpu')
        assert [t['shape'] for t in normal['actual_forward_inputs']] == [[32, 3, 640, 640], [1, 3, 640, 640]]
        committed = ['MANIFEST.json', 'COMPLETE.json'] + [str(p.relative_to(resumed)) for p in (resumed / 'batches').rglob('*.json')]
        assert all((resumed / name).read_bytes() == (straight / name).read_bytes() for name in committed)
        before_changed = (resumed / 'COMPLETE.json').read_bytes()
        changed_image = image_dir / images[0]['file_name']
        changed_image.write_bytes(changed_image.read_bytes() + b'changed-input')
        try:
            run(spec, resumed, 'cpu')
        except ValueError as error:
            assert 'Input image content changed' in str(error)
        else:
            raise AssertionError('Mutated real input bytes were not rejected')
        assert (resumed / 'COMPLETE.json').read_bytes() == before_changed
        records = [json.loads(p.read_bytes()) for p in (resumed / 'batches').glob('*/*.json') if p.name != 'COMMIT.json']
        result = {'status': 'passed', 'scope': 'Generated unlabeled images and existing preflight smoke checkpoint on CPU only',
                  'real_BDD_KITTI_inputs_used': False, 'accuracy_or_latency_results': False,
                  'images': 33, 'forced_synthetic_interruption_after_batch': 0,
                  'resumed_new_image_ids': recovery['new_image_ids'],
                  'resumed_actual_forward_inputs': recovery['actual_forward_inputs'],
                  'uninterrupted_actual_forward_inputs': normal['actual_forward_inputs'],
                  'setup_probes': {'resumed': recovery['before_or_between_batch_probes'], 'uninterrupted': normal['before_or_between_batch_probes']},
                  'committed_batch0_bytes_unchanged': True, 'complete_archive_skips_model_load_and_inference': True,
                  'resumed_and_uninterrupted_artifact_bytes_identical': True, 'artifact_files_compared': len(committed),
                  'changed_image_content_rejected_before_inference': True,
                  'all_class_predictions_retained': sum(len(r['detections']) for r in records),
                  'checkpoint_unchanged': sha256_file(a.checkpoint) == initial_checkpoint_hash,
                  'preflight_checkpoint_sha256': initial_checkpoint_hash,
                  'protocol_sha256': sha256_file(a.protocol),
                  'source_sha256': {n: sha256_file(source / n) for n in (*SOURCES, Path(__file__).name)},
                  'runtime': normal['runtime'],
                  'limits': 'CPU synthetic integration only. Actual GPU traces and all real-data checkpoints/inputs/metrics remain unevaluated.'}
        assert result['checkpoint_unchanged']
    a.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
