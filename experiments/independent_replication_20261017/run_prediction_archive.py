"""Fixed inference with content binding and immutable, resumable predictions.

No labels, AP scores, coefficient selection or training occur here. An immutable
request fixes the checkpoint, protocol, image manifest and four local sources.
Use only after the final study roster is frozen; synthetic preflight is separate.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import uuid
from pathlib import Path

from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import INFERENCE, check_class_names, image_metadata, predict_image_chunks

SOURCES = ('prediction_archive.py', 'prediction_io.py', 'fixed_geometry_predictor.py',
           'run_prediction_archive.py')


def verify_source_hashes(expected):
    if set(expected) != set(SOURCES):
        raise ValueError('Exactly the four frozen inference/archive source hashes are required')
    for name, sha in expected.items():
        if sha256_file(Path(__file__).parent / name) != sha:
            raise ValueError('Inference source changed: ' + name)


def verify_artifact(item):
    if set(item) != {'path', 'sha256'} or sha256_file(item['path']) != item['sha256']:
        raise ValueError('Artifact differs from the frozen content hash')


def verify_images(image_dir, images, check_headers=False):
    from PIL import Image
    for item in images:
        metadata = image_metadata(item)
        path = Path(image_dir) / metadata['file_name']
        if not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError('Input image content changed or missing: ' + metadata['file_name'])
        if check_headers:
            with Image.open(path) as im:
                if im.size != (metadata['width'], metadata['height']):
                    raise ValueError('Input image dimensions differ from manifest')


def runtime_binding(device):
    import torch
    from ultralytics.utils.torch_utils import init_seeds, select_device
    if importlib.metadata.version('ultralytics') != '8.2.103':
        raise ValueError('Expected pinned Ultralytics8.2.103')
    # Fixed per-process evaluation behavior. This does not alter training jobs.
    torch.set_num_threads(2)
    selected = select_device(device, verbose=False)
    init_seeds(0, deterministic=True)
    native_modules = ('ultralytics.engine.predictor', 'ultralytics.nn.autobackend',
                      'ultralytics.models.yolo.detect.predict', 'ultralytics.data.augment',
                      'ultralytics.utils.ops', 'ultralytics.nn.tasks')
    record = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n) for n in
                     ('torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'opencv-python')},
        'native_source_sha256': {n: sha256_file(importlib.import_module(n).__file__) for n in native_modules},
        'requested_device': str(device), 'selected_device': str(selected),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_cuda_build': torch.version.cuda, 'torch_num_threads': torch.get_num_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'evaluation_seed': 0,
    }
    if selected.type == 'cuda':
        prop = torch.cuda.get_device_properties(selected)
        record['gpu'] = {'name': prop.name, 'total_memory': prop.total_memory,
                         'compute_capability': [prop.major, prop.minor],
                         'uuid': str(getattr(prop, 'uuid', 'unavailable'))}
    return record


def _write_new(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())


def run(spec, output, device='cpu', max_new_batches=None, after_commit=None):
    """after_commit is a test-only fault injector; it is not exposed by the CLI."""
    from ultralytics import YOLO
    spec = json.loads(canonical(spec))
    expected_keys = {'schema', 'endpoint_id', 'dataset_id', 'checkpoint', 'protocol',
                     'image_manifest', 'image_dir', 'source_sha256'}
    if set(spec) != expected_keys or spec['schema'] != 'frozen_prediction_request_v1':
        raise ValueError('Invalid frozen inference request')
    if max_new_batches is not None and (type(max_new_batches) is not int or max_new_batches < 1):
        raise ValueError('A batch limit must be a positive integer')
    verify_source_hashes(spec['source_sha256'])
    for name in ('checkpoint', 'protocol', 'image_manifest'):
        verify_artifact(spec[name])
    protocol = json.loads(Path(spec['protocol']['path']).read_bytes())
    if protocol['evaluation']['inference'] != INFERENCE:
        raise ValueError('Inference settings do not match the frozen protocol')
    images = json.loads(Path(spec['image_manifest']['path']).read_bytes())
    runtime = runtime_binding(device)
    bindings = {'endpoint_id': spec['endpoint_id'], 'dataset_id': spec['dataset_id'],
                'checkpoint_sha256': spec['checkpoint']['sha256'],
                'protocol_sha256': spec['protocol']['sha256'],
                'source_sha256': {**spec['source_sha256'], 'frozen_request': digest(canonical(spec))},
                'runtime': runtime, 'inference': INFERENCE}
    output = Path(output)
    journal_dir = output.parent / (output.name + '_executions')
    journal_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    journal = {'request_sha256': digest(canonical(spec)), 'status': 'started',
               'new_batch_indices': [], 'new_image_ids': [], 'before_or_between_batch_probes': [],
               'actual_forward_inputs': [], 'checkpoint_loaded': False, 'runtime': runtime}
    _write_new(journal_dir / (run_id + '_started.json'), journal)
    handle = None
    try:
        with PredictionArchive(output, bindings, images) as archive:
            verify_images(spec['image_dir'], images, check_headers=True)
            state = archive.verify()
            journal['previous_committed_batches'] = state['committed_batches']
            if not state['complete']:
                model = YOLO(spec['checkpoint']['path'])
                journal['checkpoint_loaded'] = True
                check_class_names(model.names)
                actual, outside = [], []
                active_batch = False

                def capture(module, inputs):
                    tensor = inputs[0]
                    item = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                            'device': str(tensor.device)}
                    (actual if active_batch else outside).append(item)

                def batch_start(predictor):
                    nonlocal active_batch
                    if active_batch:
                        raise ValueError('Nested prediction batch callbacks')
                    active_batch = True

                def batch_end(predictor):
                    nonlocal active_batch
                    active_batch = False

                handle = model.model.register_forward_pre_hook(capture)
                model.add_callback('on_predict_batch_start', batch_start)
                model.add_callback('on_predict_batch_end', batch_end)
                for index in range(state['committed_batches'], archive.batch_count):
                    batch_images = archive.expected_images(index)
                    verify_images(spec['image_dir'], batch_images)
                    actual.clear()
                    batches = list(predict_image_chunks(model, spec['image_dir'], batch_images, protocol, device))
                    if active_batch or len(batches) != 1:
                        raise ValueError('Expected one completed explicit prediction batch')
                    if any(t['device'] != runtime['selected_device'] for t in actual):
                        raise ValueError('Actual inference device differs from frozen runtime')
                    verify_images(spec['image_dir'], batch_images)
                    archive.commit(index, batches[0], actual)
                    journal['new_batch_indices'].append(index)
                    journal['new_image_ids'].extend(im['id'] for im in batch_images)
                    journal['actual_forward_inputs'].extend(json.loads(canonical(actual)))
                    journal['before_or_between_batch_probes'] = json.loads(canonical(outside))
                    if after_commit is not None:
                        after_commit(index)
                    if max_new_batches is not None and len(journal['new_batch_indices']) >= max_new_batches:
                        break
                handle.remove()
                handle = None
                # Full before/after content verification catches accidental changes
                # beyond the active batch as well as checkpoint/source mutation.
                verify_images(spec['image_dir'], images)
                for name in ('checkpoint', 'protocol', 'image_manifest'):
                    verify_artifact(spec[name])
                verify_source_hashes(spec['source_sha256'])
                state = archive.verify()
                if state['committed_batches'] == archive.batch_count:
                    state = archive.finalize()
            journal['status'] = 'complete' if state['complete'] else 'incomplete_batch_limit'
            journal['committed_images'] = state['committed_images']
            journal['archive_complete'] = state['complete']
            journal['manifest_sha256'] = archive.manifest_sha256
        _write_new(journal_dir / (run_id + '_finished.json'), journal)
        return journal
    except BaseException as error:
        journal['status'] = 'failed'
        journal['error_type'] = type(error).__name__
        journal['error'] = str(error)
        _write_new(journal_dir / (run_id + '_failed.json'), journal)
        raise
    finally:
        if handle is not None:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-batches', type=int)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_bytes()), args.output, args.device, args.max_new_batches)
    print(json.dumps({k: result[k] for k in ('status', 'committed_images', 'new_batch_indices', 'archive_complete')}, indent=2))


if __name__ == '__main__':
    main()
