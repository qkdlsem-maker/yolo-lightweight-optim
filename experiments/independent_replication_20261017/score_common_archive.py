"""Score a complete immutable prediction archive under every fixed label view.

CPU-only Linux scorer; no model loading, inference, threshold selection or data
mutation. KITTI's two prespecified common-metric variants are an atomic unit.
The distinct native BDD and KITTI metrics are not computed by this module.
"""
import argparse
import fcntl
import json
import os
import uuid
from pathlib import Path

from common_ap_core import evaluate, runtime_record
from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import common_rows, image_metadata

SOURCES = ('common_ap_core.py', 'prediction_archive.py', 'prediction_io.py',
           'score_common_archive.py', 'common_annotation_policy.json')
VARIANTS = {'bdd': ('bdd_common',),
            'kitti': ('kitti_source_ignore', 'kitti_no_ignore_sensitivity')}


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


def check_sources(expected):
    if set(expected) != set(SOURCES):
        raise ValueError('All five fixed scoring source/policy hashes are required')
    for name in SOURCES:
        if sha256_file(Path(__file__).parent / name) != expected[name]:
            raise ValueError('Scoring source changed: ' + name)


def verify_output(output, manifest):
    variants = VARIANTS[manifest['request']['dataset']]
    names = {v + '.json' for v in variants} | {'MANIFEST.json', 'COMPLETE.json'}
    if {p.name for p in output.iterdir()} != names or any(p.is_symlink() for p in output.iterdir()):
        raise ValueError('Missing or extra score output file')
    manifest_raw = (output / 'MANIFEST.json').read_bytes()
    if manifest_raw != canonical(manifest):
        raise ValueError('Existing score request, prediction archive or runtime differs')
    complete = json.loads((output / 'COMPLETE.json').read_bytes())
    expected = {'schema': 'common_archive_scores_complete_v1',
                'manifest_sha256': digest(manifest_raw),
                'outputs': {v + '.json': sha256_file(output / (v + '.json')) for v in variants}}
    if complete != expected:
        raise ValueError('Score output bytes changed')
    return {'status': 'complete', 'new_scores_computed': False,
            'variants': list(variants), 'manifest_sha256': digest(manifest_raw)}


def run(request, output):
    request = json.loads(canonical(request))
    required = {'schema', 'dataset', 'archive_path', 'archive_manifest_sha256',
                'archive_complete_sha256', 'source_sha256', 'annotation_views'}
    if set(request) != required or request['schema'] != 'common_archive_scoring_request_v1' or request['dataset'] not in VARIANTS:
        raise ValueError('Invalid scoring request')
    variants = VARIANTS[request['dataset']]
    if set(request['annotation_views']) != set(variants):
        raise ValueError('Every fixed annotation variant must be scored together')
    check_sources(request['source_sha256'])
    folder = Path(request['archive_path'])
    for name, key in [('MANIFEST.json', 'archive_manifest_sha256'), ('COMPLETE.json', 'archive_complete_sha256')]:
        if sha256_file(folder / name) != request[key]:
            raise ValueError('Frozen prediction archive binding changed')
    archived = json.loads((folder / 'MANIFEST.json').read_bytes())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with PredictionArchive(folder, archived['bindings'], archived['images']) as archive:
            if not archive.verify()['complete']:
                raise ValueError('Scoring requires complete verified predictions')
            expected_images = [image_metadata(im) for im in archive.images]
            views = {}
            for variant in variants:
                item = request['annotation_views'][variant]
                if set(item) != {'path', 'sha256'} or sha256_file(item['path']) != item['sha256']:
                    raise ValueError('Annotation view changed')
                data = json.loads(Path(item['path']).read_bytes())
                if data['info']['annotation_variant'] != variant or data['info']['policy_sha256'] != request['source_sha256']['common_annotation_policy.json']:
                    raise ValueError('Wrong annotation variant/policy binding')
                if [image_metadata(im) for im in data['images']] != expected_images:
                    raise ValueError('Annotation and prediction image populations/order differ')
                if data['categories'] != [{'id': 0, 'name': 'person'}, {'id': 2, 'name': 'car'}]:
                    raise ValueError('Expected the fixed person/car category mapping')
                views[variant] = data
            if request['dataset'] == 'kitti':
                ordinary = [{k: v for k, v in row.items() if k != 'id'}
                            for row in views['kitti_source_ignore']['annotations'] if row['iscrowd'] == 0]
                strict = [{k: v for k, v in row.items() if k != 'id'}
                          for row in views['kitti_no_ignore_sensitivity']['annotations']]
                if ordinary != strict or any(row['iscrowd'] != 0 for row in strict):
                    raise ValueError('KITTI variants must have exactly identical ordinary targets')
            manifest = {'schema': 'common_archive_scoring_manifest_v1', 'request': request,
                        'runtime': runtime_record(), 'prediction_bindings': archived['bindings']}
            if output.exists():
                return verify_output(output, manifest)
            # One projection from one verified all-class archive supplies all
            # variants, including the same empty-image population.
            predictions = []
            all_count = 0
            for record in archive.iter_records():
                all_count += len(record['detections'])
                predictions.extend(common_rows(record))
            projection_hash = digest(canonical(predictions))
            stage = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
            stage.mkdir()
            _write(stage / 'MANIFEST.json', canonical(manifest))
            for variant in variants:
                data = views[variant]
                score = evaluate(data['images'], data['annotations'], predictions, include_matches=False)
                artifact = {'schema': 'common_archive_score_v1', 'annotation_variant': variant,
                            'annotation_sha256': request['annotation_views'][variant]['sha256'],
                            'archive_manifest_sha256': request['archive_manifest_sha256'],
                            'archive_complete_sha256': request['archive_complete_sha256'],
                            'shared_common_prediction_projection_sha256': projection_hash,
                            'all_class_predictions_preserved': all_count,
                            'mapped_person_car_predictions': len(predictions), 'score': score}
                _write(stage / (variant + '.json'), canonical(artifact))
            # Do not publish a successful set if any pinned input changes while
            # scoring. Partial directories remain evidence on technical failure.
            check_sources(request['source_sha256'])
            archive.verify()
            for item in request['annotation_views'].values():
                if sha256_file(item['path']) != item['sha256']:
                    raise ValueError('Annotation bytes changed during scoring')
            complete = {'schema': 'common_archive_scores_complete_v1',
                        'manifest_sha256': digest(canonical(manifest)),
                        'outputs': {v + '.json': sha256_file(stage / (v + '.json')) for v in variants}}
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
