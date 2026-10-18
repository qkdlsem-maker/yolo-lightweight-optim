"""Run the pinned, previously verified native KITTI 2D core on an archive.

CPU only. The input predictions are complete and immutable; no inference,
model selection, threshold changes or upstream scoring-code changes occur.
All outputs, including private label snapshots, stay below a new output path.
"""
import argparse
import fcntl
import json
import math
import os
import platform
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import kitti_text

SOURCES = ('prediction_archive.py', 'prediction_io.py', 'score_native_kitti_archive.py')
CORE = {
    'executable': '4524fb43a3b39662140f1f41a749d3cf42e395e18cd23420260fa796d89b64c9',
    'original_source': 'b16410ba8914a732405ec937a16d1cd039b6e3bf7d9e8edb6dd3ec79c15892b4',
    'adapted_source': '20c521ef2b30b37f34b52cdbfee3c6e7daa0d3650a603c2b209a5dd39f91c45e',
    'build_manifest': '427e55b18221e0c67788c0b81240d1b31f20d0e77eb6b3f3c6eba7339ee28a33',
    'prior_synthetic_proof': '444b92667b3656ccba487678ffbf088d03978fb3354fcd1a01ea88eff7ec3527',
}
CLASSES = ('car', 'pedestrian')
DIFFICULTIES = ('easy', 'moderate', 'hard')


def _write(path, raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_inputs(request):
    expected = request['source_sha256']
    if set(expected) != set(SOURCES):
        raise ValueError('All fixed scoring source hashes are required')
    for name in SOURCES:
        if sha256_file(Path(__file__).parent / name) != expected[name]:
            raise ValueError('Scoring source changed: ' + name)
    if set(request['native_core_paths']) != set(CORE):
        raise ValueError('All original-core provenance files are required')
    for name, expected_hash in CORE.items():
        if sha256_file(request['native_core_paths'][name]) != expected_hash:
            raise ValueError('Pinned native core changed: ' + name)
    spec = request['label_zip']
    if set(spec) != {'path', 'sha256'} or sha256_file(spec['path']) != spec['sha256']:
        raise ValueError('Label archive bytes changed')
    archive = Path(request['archive_path'])
    for name, key in [('MANIFEST.json', 'archive_manifest_sha256'), ('COMPLETE.json', 'archive_complete_sha256')]:
        if sha256_file(archive / name) != request[key]:
            raise ValueError('Prediction archive binding changed')


def label_snapshot(zip_path, images):
    """Read only exact expected members; never extract paths from a ZIP."""
    labels = {}
    names = [f'training/label_2/{i:06d}.txt' for i in range(len(images))]
    if [x['file_name'] for x in images] != [f'{i:06d}.png' for i in range(len(images))]:
        raise ValueError('Native KITTI requires the complete ordered zero-based frame population')
    if [x['id'] for x in images] != list(range(1, len(images) + 1)):
        raise ValueError('Unexpected archive image ID order')
    with zipfile.ZipFile(zip_path) as z:
        files = [info.filename for info in z.infolist() if not info.is_dir()]
        if len(files) != len(set(files)) or set(files) != set(names):
            raise ValueError('Label ZIP and prediction image populations differ')
        for name in names:
            labels[Path(name).name] = z.read(name)
    return labels


def parse_curves(folder):
    result = {}
    if {p.name for p in folder.iterdir()} != {f'stats_{c}_detection.txt' for c in CLASSES}:
        raise ValueError('Unexpected native evaluator output set')
    for category in CLASSES:
        rows = [[float(x) for x in line.split()]
                for line in (folder / f'stats_{category}_detection.txt').read_text().splitlines()]
        if len(rows) != 3 or any(len(row) != 41 for row in rows):
            raise ValueError('Native precision curves must have shape 3 by 41')
        if not all(math.isfinite(x) and 0 <= x <= 1 for row in rows for x in row):
            raise ValueError('Invalid native precision value')
        result[category] = {difficulty: {'precision_41': row, 'AP_R40': math.fsum(row[1:]) / 40}
                            for difficulty, row in zip(DIFFICULTIES, rows)}
    return result


def inventory(folder):
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Output symlinks are not permitted')
        if path.is_file() and path != folder / 'COMPLETE.json':
            files[path.relative_to(folder).as_posix()] = sha256_file(path)
    return files


def verify_output(output, manifest):
    if (output / 'MANIFEST.json').read_bytes() != canonical(manifest):
        raise ValueError('Existing request, predictions or runtime differs')
    completion = json.loads((output / 'COMPLETE.json').read_bytes())
    expected = {'schema': 'native_kitti_archive_scores_complete_v1',
                'manifest_sha256': digest(canonical(manifest)), 'files': inventory(output)}
    if completion != expected:
        raise ValueError('Native scoring output bytes changed')
    return {'status': 'complete', 'new_scores_computed': False,
            'manifest_sha256': expected['manifest_sha256']}


def run(request, output):
    request = json.loads(canonical(request))
    required = {'schema', 'archive_path', 'archive_manifest_sha256', 'archive_complete_sha256',
                'source_sha256', 'native_core_paths', 'label_zip'}
    if set(request) != required or request['schema'] != 'native_kitti_archive_scoring_request_v1':
        raise ValueError('Invalid native KITTI scoring request')
    check_inputs(request)
    folder = Path(request['archive_path'])
    archived = json.loads((folder / 'MANIFEST.json').read_bytes())
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with PredictionArchive(folder, archived['bindings'], archived['images']) as archive:
            if not archive.verify()['complete']:
                raise ValueError('Native scoring requires complete verified predictions')
            labels = label_snapshot(request['label_zip']['path'], archive.images)
            manifest = {'schema': 'native_kitti_archive_scoring_manifest_v1', 'request': request,
                        'core_sha256': CORE, 'prediction_bindings': archived['bindings'],
                        'runtime': {'python': sys.version, 'platform': platform.platform()},
                        'images': len(archive.images),
                        'label_member_sha256': {n: digest(raw) for n, raw in labels.items()}}
            if output.exists():
                return verify_output(output, manifest)
            stage = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
            stage.mkdir()
            _write(stage / 'MANIFEST.json', canonical(manifest))
            for name in ('ground_truth', 'predictions', 'curves'):
                (stage / name).mkdir()
            # Preserve exactly the pinned ZIP bytes; no writes to original data.
            for name, raw in labels.items():
                _write(stage / 'ground_truth' / name, raw)
            all_count = mapped_count = empty_count = zero_area = 0
            for record in archive.iter_records():
                raw = kitti_text(record).encode('ascii')
                all_count += len(record['detections'])
                mapped_count += raw.count(b'\n')
                empty_count += not raw
                zero_area += sum((x2 == x1 or y2 == y1) and c in (0, 2)
                                 for x1, y1, x2, y2, score, c in record['detections'])
                name = Path(record['image']['file_name']).stem + '.txt'
                _write(stage / 'predictions' / name, raw)
            _sync(stage / 'ground_truth')
            _sync(stage / 'predictions')
            snapshot_hashes = {name: inventory(stage / name) for name in ('ground_truth', 'predictions')}
            executable = str(Path(request['native_core_paths']['executable']).resolve())
            command = [executable, str(stage / 'ground_truth'), str(stage / 'predictions'),
                       str(stage / 'curves'), str(len(archive.images))]
            with (stage / 'evaluator.stdout.txt').open('xb') as stdout, (stage / 'evaluator.stderr.txt').open('xb') as stderr:
                subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=1800)
                for stream in (stdout, stderr):
                    stream.flush()
                    os.fsync(stream.fileno())
            scores = parse_curves(stage / 'curves')
            if snapshot_hashes != {name: inventory(stage / name) for name in snapshot_hashes}:
                raise ValueError('Label or prediction text snapshot changed during native scoring')
            score = {'schema': 'native_kitti_archive_score_v1', 'AP_scale': 'fraction_0_to_1',
                     'classes': scores,
                     'mean_moderate_AP_R40': math.fsum(scores[c]['moderate']['AP_R40'] for c in CLASSES) / 2,
                     'AP_R40_definition': 'Arithmetic mean of precision slots 1 through 40; slot 0 excluded',
                     'images': len(archive.images), 'all_class_predictions_preserved': all_count,
                     'mapped_predictions_exported': mapped_count, 'empty_prediction_files': empty_count,
                     'zero_area_mapped_predictions_exported': zero_area,
                     'archive_manifest_sha256': request['archive_manifest_sha256'],
                     'archive_complete_sha256': request['archive_complete_sha256'],
                     'snapshot_sha256': snapshot_hashes,
                     'native_core_sha256': CORE,
                     'label_zip_sha256': request['label_zip']['sha256']}
            _write(stage / 'score.json', canonical(score))
            check_inputs(request)
            archive.verify()
            # Snapshot and text hashes are saved alongside all precision curves.
            complete = {'schema': 'native_kitti_archive_scores_complete_v1',
                        'manifest_sha256': digest(canonical(manifest)), 'files': inventory(stage)}
            for path in (stage / 'curves').iterdir():
                with path.open('rb') as f:
                    os.fsync(f.fileno())
            _sync(stage / 'curves')
            _write(stage / 'COMPLETE.json', canonical(complete))
            _sync(stage)
            os.rename(stage, output)
            _sync(output.parent)
            result = verify_output(output, manifest)
            result['new_scores_computed'] = True
            return result
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(run(json.loads(args.request.read_bytes()), args.output), indent=2))


if __name__ == '__main__':
    main()
