"""Audited standalone native ten-class validation, isolated from training data.

The subclass only checks inputs and records native statistics. It does not
change preprocessing, postprocessing, matching or AP integration. Successful
outputs are immutable; incomplete attempts remain separate for inspection.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path

from prediction_archive import canonical, digest, sha256_file
from prediction_io import INFERENCE, check_class_names, image_metadata
from run_prediction_archive import runtime_binding, verify_artifact, verify_images

SOURCES = ('prediction_archive.py', 'prediction_io.py', 'run_prediction_archive.py',
           'run_native_bdd_validation.py')
NATIVE_MODULES = ('ultralytics.engine.validator', 'ultralytics.models.yolo.detect.val',
                  'ultralytics.data.dataset', 'ultralytics.data.base', 'ultralytics.data.build',
                  'ultralytics.data.utils', 'ultralytics.utils.metrics', 'ultralytics.cfg')


def write_new(path, value):
    raw = value if isinstance(value, bytes) else canonical(value)
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def inventory(folder):
    result = {}
    for path in sorted(folder.rglob('*')):
        key = path.relative_to(folder).as_posix()
        if path.is_symlink():
            result[key] = {'symlink_target': os.readlink(path)}
        elif path.is_file() and key != 'COMPLETE.json':
            result[key] = {'sha256': sha256_file(path)}
    return result


def source_check(expected):
    if set(expected) != set(SOURCES):
        raise ValueError('Every fixed source hash is required')
    for name, sha in expected.items():
        if sha256_file(Path(__file__).parent / name) != sha:
            raise ValueError('Validation source changed: ' + name)


def native_hashes():
    from ultralytics.utils import DEFAULT_CFG_PATH
    return {**{n: sha256_file(importlib.import_module(n).__file__) for n in NATIVE_MODULES},
            'default_cfg_yaml': sha256_file(DEFAULT_CFG_PATH)}


def verify_labels(directory, images, expected):
    names = [Path(im['file_name']).stem + '.txt' for im in images]
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError('Expected unique sorted image stems')
    directory = Path(directory)
    if {p.name for p in directory.glob('*.txt')} != set(names):
        raise ValueError('Label and image populations differ')
    h = hashlib.sha256()
    labels = {}
    for name in names:
        raw = (directory / name).read_bytes()
        h.update(name.encode() + b'\0' + raw)
        labels[name] = raw
    if h.hexdigest() != expected:
        raise ValueError('Ordered label bytes differ from the fixed manifest')
    return labels


def verify_complete(output, manifest):
    if (output / 'MANIFEST.json').read_bytes() != canonical(manifest):
        raise ValueError('Completed request or runtime differs')
    expected = {'schema': 'native_bdd_validation_complete_v1',
                'manifest_sha256': digest(canonical(manifest)), 'files': inventory(output)}
    if json.loads((output / 'COMPLETE.json').read_bytes()) != expected:
        raise ValueError('Completed validation artifacts changed')
    return {'status': 'complete', 'new_inference': False,
            'manifest_sha256': expected['manifest_sha256']}


def run(request, output, device='cpu'):
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    request = json.loads(canonical(request))
    required = {'schema', 'endpoint_id', 'checkpoint', 'protocol', 'image_manifest',
                'image_dir', 'label_dir', 'dataset_manifest', 'data_yaml', 'source_sha256'}
    if set(request) != required or request['schema'] != 'native_bdd_validation_request_v1':
        raise ValueError('Invalid native validation request')
    source_check(request['source_sha256'])
    for name in ('checkpoint', 'protocol', 'image_manifest', 'dataset_manifest', 'data_yaml'):
        verify_artifact(request[name])
    protocol = json.loads(Path(request['protocol']['path']).read_bytes())
    if protocol['evaluation']['inference'] != INFERENCE:
        raise ValueError('Inference configuration differs from protocol')
    if protocol['data']['yaml_sha256'] != request['data_yaml']['sha256'] or protocol['data']['manifest_sha256'] != request['dataset_manifest']['sha256']:
        raise ValueError('Data provenance is not bound to protocol')
    images = json.loads(Path(request['image_manifest']['path']).read_bytes())
    if len(images) != protocol['data']['validation_images'] or not images:
        raise ValueError('Unexpected validation population')
    if [im['id'] for im in images] != list(range(1, len(images) + 1)):
        raise ValueError('Image IDs/order differ')
    if [im['file_name'] for im in images] != sorted(im['file_name'] for im in images):
        raise ValueError('Native validation requires the frozen filename order')
    data_manifest = json.loads(Path(request['dataset_manifest']['path']).read_bytes())
    if data_manifest['splits']['val']['images'] != len(images):
        raise ValueError('Label manifest image count differs')
    expected_labels = data_manifest['splits']['val']['ordered_label_content_sha256']
    labels = verify_labels(request['label_dir'], images, expected_labels)
    verify_images(request['image_dir'], images, check_headers=True)
    names = yaml.safe_load(Path(request['data_yaml']['path']).read_bytes())['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    check_class_names(names)
    runtime = runtime_binding(device)
    native = native_hashes()
    manifest = {'schema': 'native_bdd_validation_manifest_v1', 'request': request,
                'runtime': runtime, 'native_validation_source_sha256': native}
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle = None
    try:
        if output.exists():
            return verify_complete(output, manifest)
        stage = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
        stage.mkdir()
        write_new(stage / 'MANIFEST.json', manifest)
        for name in ('view/images/val', 'view/labels/val', 'native_stats'):
            (stage / name).mkdir(parents=True)
        for im in images:
            target = (Path(request['image_dir']) / im['file_name']).absolute()
            (stage / 'view/images/val' / im['file_name']).symlink_to(target)
        for name, raw in labels.items():
            write_new(stage / 'view/labels/val' / name, raw)
        yaml_path = stage / 'view/data.yaml'
        write_new(yaml_path, yaml.safe_dump({'path': str(stage / 'view'), 'train': 'images/val',
                                          'val': 'images/val', 'names': names}).encode())
        audit = {'images': [], 'actual_forward_inputs': [], 'setup_probes': [], 'batches': [],
                 'native_loaded_labels_match_raw_float32_unique_rows': False}
        active = False

        class AuditedValidator(DetectionValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                if self.is_coco or self.is_lvis or self.training or self.args.half:
                    raise ValueError('Expected native BDD standalone FP32 validation')

            def preprocess(self, batch):
                start = len(audit['images'])
                expected = images[start:start + len(batch['im_file'])]
                if [Path(p).name for p in batch['im_file']] != [im['file_name'] for im in expected]:
                    raise ValueError('Actual validation batch image order differs')
                if [tuple(v) for v in batch['ori_shape']] != [(im['height'], im['width']) for im in expected]:
                    raise ValueError('Actual original image dimensions differ')
                verify_images(request['image_dir'], expected)
                batch = super().preprocess(batch)
                audit['images'].extend(im['id'] for im in expected)
                return batch

            def update_metrics(self, preds, batch):
                previous = len(self.stats['tp'])
                super().update_metrics(preds, batch)
                expected = images[len(audit['images']) - len(preds):len(audit['images'])]
                ids, target_ids, ledger = [], [], []
                for si, (pred, im) in enumerate(zip(preds, expected)):
                    nl = int((batch['batch_idx'] == si).sum().item())
                    npred = len(pred)
                    ledger.append({'image_id': im['id'], 'predictions': npred, 'targets': nl})
                    if npred or nl:
                        ids.extend([im['id']] * npred)
                        target_ids.extend([im['id']] * nl)
                rows = {k: v[previous:] for k, v in self.stats.items()}
                if len(rows['tp']) != sum(x['predictions'] > 0 or x['targets'] > 0 for x in ledger):
                    raise ValueError('Native per-image statistic alignment differs')
                values = {k: torch.cat(v, 0).cpu().numpy() if v else np.empty((0, 10) if k == 'tp' else (0,))
                          for k, v in rows.items()}
                if len(values['conf']) != len(ids) or len(values['target_cls']) != len(target_ids):
                    raise ValueError('Native statistic row count differs')
                values.update(prediction_image_id=np.asarray(ids, dtype=np.int64),
                              target_image_id=np.asarray(target_ids, dtype=np.int64))
                if not all(np.isfinite(a).all() for a in values.values()):
                    raise ValueError('Nonfinite native statistic')
                np.savez_compressed(stage / 'native_stats' / f'batch_{self.batch_i:04d}.npz', **values)
                audit['batches'].append({'batch_index': self.batch_i, 'images': ledger})

        model = YOLO(request['checkpoint']['path'])
        if model.names != names:
            raise ValueError('Model class names differ from fixed data configuration')

        def capture(module, inputs):
            t = inputs[0]
            trace = {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device)}
            if active:
                if t.dtype != torch.float32 or list(t.shape[1:]) != [3, 640, 640] or str(t.device) != runtime['selected_device']:
                    raise ValueError('Actual native validation geometry/precision/device differs')
                audit['actual_forward_inputs'].append(trace)
            else:
                audit['setup_probes'].append(trace)

        def start(validator):
            dataset = validator.dataloader.dataset
            if dataset.rect or validator.dataloader.batch_size != 32:
                raise ValueError('Unexpected native loader configuration')
            if [Path(p).name for p in dataset.im_files] != [im['file_name'] for im in images]:
                raise ValueError('Native loader removed or reordered images')
            loaded_boxes = 0
            for im, item in zip(images, dataset.labels):
                raw = labels[Path(im['file_name']).stem + '.txt']
                rows = np.asarray([line.split() for line in raw.decode().splitlines() if line.strip()], dtype=np.float32).reshape(-1, 5)
                expected = np.unique(rows, axis=0)
                actual = np.concatenate((item['cls'], item['bboxes']), axis=1)
                if not item['normalized'] or item['bbox_format'] != 'xywh' or item['segments']:
                    raise ValueError('Expected normalized detection boxes only')
                if len(actual) != len(expected) or not np.array_equal(np.unique(actual, axis=0), expected):
                    raise ValueError('Native loaded labels differ from pinned raw float32 rows')
                loaded_boxes += len(actual)
            audit.update(native_loaded_labels_match_raw_float32_unique_rows=True,
                         native_loaded_boxes=loaded_boxes, raw_label_rows=sum(len(raw.splitlines()) for raw in labels.values()),
                         workers=validator.args.workers, training_context=validator.training)
            if audit['raw_label_rows'] != data_manifest['splits']['val']['boxes']:
                raise ValueError('Raw label row count differs from data manifest')

        def batch_start(validator):
            nonlocal active
            active = True

        def batch_end(validator):
            nonlocal active
            active = False

        handle = model.model.register_forward_pre_hook(capture)
        model.add_callback('on_val_start', start)
        model.add_callback('on_val_batch_start', batch_start)
        model.add_callback('on_val_batch_end', batch_end)
        metrics = model.val(validator=AuditedValidator, data=str(yaml_path), imgsz=640, batch=32,
                            device=device, workers=4, half=False, rect=False, conf=.001, iou=.7,
                            max_det=300, augment=False, single_cls=False, classes=None, agnostic_nms=False,
                            plots=False, verbose=False, save_json=False, save_txt=False, save_hybrid=False,
                            project=str(stage), name='native', exist_ok=False)
        handle.remove()
        handle = None
        expected_shapes = [[min(32, len(images) - i), 3, 640, 640] for i in range(0, len(images), 32)]
        if active or audit['images'] != [im['id'] for im in images] or [t['shape'] for t in audit['actual_forward_inputs']] != expected_shapes:
            raise ValueError('Incomplete or repeated native validation forward pass')
        score = {'schema': 'native_bdd_validation_score_v1', 'results': metrics.results_dict,
                 'class_AP_by_IoU': np.asarray(metrics.box.all_ap).tolist(),
                 'AP_class_indices': np.asarray(metrics.box.ap_class_index).tolist(),
                 'IoU_thresholds': [round(.5 + .05 * i, 2) for i in range(10)],
                 'images': len(images), 'native_boxes': audit['native_loaded_boxes'],
                 'metric_definition': 'Unchanged Ultralytics8.2.103 native detection matching and AP integration; ten BDD classes',
                 'inference': INFERENCE, 'speed_not_a_controlled_latency_benchmark': metrics.speed}
        write_new(stage / 'score.json', score)
        write_new(stage / 'input_audit.json', audit)
        verify_images(request['image_dir'], images)
        verify_labels(request['label_dir'], images, expected_labels)
        verify_labels(stage / 'view/labels/val', images, expected_labels)
        for name in ('checkpoint', 'protocol', 'image_manifest', 'dataset_manifest', 'data_yaml'):
            verify_artifact(request[name])
        source_check(request['source_sha256'])
        if native_hashes() != native:
            raise ValueError('Native library source changed during validation')
        write_new(stage / 'COMPLETE.json', {'schema': 'native_bdd_validation_complete_v1',
                                          'manifest_sha256': digest(canonical(manifest)), 'files': inventory(stage)})
        # Flush generated cache/statistic files and directories before publication.
        for path in [*stage.rglob('*'), stage]:
            if path.is_symlink():
                continue
            fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.rename(stage, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = verify_complete(output, manifest)
        result['new_inference'] = True
        return result
    finally:
        if handle is not None:
            handle.remove()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(json.dumps(run(json.loads(a.request.read_bytes()), a.output, a.device), indent=2))


if __name__ == '__main__':
    main()
