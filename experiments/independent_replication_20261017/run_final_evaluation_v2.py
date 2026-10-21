"""Physical GPU CLI binding; immutable completed-output recovery only."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def immutable(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(obj)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Refusing to replace differing artifact: '+str(path))
        return
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def artifact(ref):
    if set(ref) != {'path', 'sha256'} or sha(ref['path']) != ref['sha256']:
        raise ValueError('Frozen artifact differs: '+str(ref))


def validate_plan(plan):
    if plan['schema'] != 'fixed_evaluation_execution_plan_v1':
        raise ValueError('Unexpected plan schema')
    artifact(plan['roster'])
    artifact(plan['protocol'])
    roster = read(plan['roster']['path'])
    protocol = read(plan['protocol']['path'])
    root = Path(plan['protocol']['path']).parent
    if (root/'PIPELINE_FAILURE.json').exists():
        raise ValueError('Training failure requires investigation')
    marker = read(root/'TRAINING_COMPLETE.json')
    assert marker['status'] == 'complete' and marker['base_runs'] == 6 and marker['continuation_runs'] == 21
    assert roster['protocol_sha256'] == plan['protocol']['sha256']
    for name, digest in roster['audit_sha256'].items():
        path = root/'evaluation_audits'/name
        assert sha(path) == digest, name
        assert read(path)['status'] == 'passed', name
    assert len(roster['endpoints']) == len(plan['jobs']) == 48
    expected = set()
    for seed in protocol['pair_seeds']:
        for arch in ['nano', 'large']:
            expected.add(f'base_{arch}_seed{seed}_e100')
        for arm, w in [('control', 0.)]+[(a, w) for a in ['mse', 'cwd'] for w in protocol['continuation']['positive_weights']]:
            for e in [20, 40]:
                expected.add(f'pair{seed}_{arm}_w{w:g}_e{e}')
    assert expected == {x['endpoint_id'] for x in roster['endpoints']} == {x['endpoint_id'] for x in plan['jobs']}
    assert len({x['checkpoint']['path'] for x in roster['endpoints']}) == 48
    assert plan['execution']['lanes'] == 2 and plan['execution']['physical_gpu_by_lane'] == [0, 1]
    output = Path(plan['output_root']).resolve()
    assert output.parent == root.resolve() and output.name == 'final_evaluation'
    refs = {}
    for i, (row, job) in enumerate(zip(roster['endpoints'], plan['jobs'])):
        assert row['endpoint_id'] == job['endpoint_id'] and job['lane'] == i % 2
        assert Path(job['output']).resolve() == output/'endpoints'/row['endpoint_id']
        for key, spec in [('native_bdd', job['native_bdd_request']), *job['prediction_requests'].items()]:
            assert spec['endpoint_id'] == row['endpoint_id']
            assert spec['checkpoint'] == row['checkpoint'] and spec['protocol'] == plan['protocol']
            for name in ['checkpoint', 'protocol', 'image_manifest', 'dataset_manifest', 'data_yaml']:
                if name in spec:
                    ref = spec[name]
                    if ref['path'] in refs:
                        assert refs[ref['path']] == ref['sha256']
                    refs[ref['path']] = ref['sha256']
            assert all(plan['source_sha256'][n] == h for n, h in spec['source_sha256'].items())
            if key != 'native_bdd':
                assert spec['dataset_id'] == key and key in ['bdd', 'kitti']
        assert set(job['prediction_requests']) == {'bdd', 'kitti'}
    for path, digest in refs.items():
        assert sha(path) == digest, path
    for name, digest in plan['source_sha256'].items():
        assert sha(Path(plan['evaluation_tools'])/name) == digest, name
    for ref in plan['annotation_views'].values():
        artifact(ref)
    artifact(plan['label_zip'])
    # Load already-validated CPU scoring modules only; no model or score access.
    sys.path.insert(0, str(Path(plan['evaluation_tools']).resolve()))
    from score_native_kitti_archive import CORE
    for name, digest in CORE.items():
        assert sha(plan['native_core_paths'][name]) == digest, name
    import importlib.metadata
    assert importlib.metadata.version('pycocotools') == '2.0.11'
    assert importlib.metadata.version('ultralytics') == '8.2.103'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    devices = [line.split(',') for line in gpu.strip().splitlines()]
    assert len(devices) >= 2 and [int(x[0]) for x in devices[:2]] == [0, 1]
    assert all(int(x[2]) < 1024 for x in devices[:2]), 'GPU already in use; do not overlap work'
    free = shutil.disk_usage(root).free
    assert free >= plan['execution']['minimum_free_GiB_before_start']*1024**3
    return {'status': 'passed', 'utc': utc(), 'endpoints': 48, 'gpu_dataset_passes': 144,
            'audit_files_verified': len(roster['audit_sha256']), 'bound_artifacts_verified': len(refs),
            'source_files_verified': len(plan['source_sha256']), 'free_bytes': free,
            'gpu_inventory': gpu.strip().splitlines(), 'scope': 'No model inference or scoring. All prespecified endpoint and source hashes verified.'}


def scoring_request(plan, job, dataset, native=False):
    archive = Path(job['output'])/(dataset+'_predictions')
    request = {'schema': 'native_kitti_archive_scoring_request_v1' if native else 'common_archive_scoring_request_v1',
               'archive_path': str(archive), 'archive_manifest_sha256': sha(archive/'MANIFEST.json'),
               'archive_complete_sha256': sha(archive/'COMPLETE.json')}
    if native:
        request.update(source_sha256=plan['kitti_source_sha256'], native_core_paths=plan['native_core_paths'], label_zip=plan['label_zip'])
    else:
        names = ['bdd_common'] if dataset == 'bdd' else ['kitti_source_ignore', 'kitti_no_ignore_sensitivity']
        request.update(dataset=dataset, source_sha256=plan['common_source_sha256'],
                       annotation_views={n: plan['annotation_views'][n] for n in names})
    return request



def step_device(plan, job, name):
    device = str(job['lane'])
    override = plan.get('_recovery', {}).get('completed_device_overrides', {}).get(job['endpoint_id']+'/'+name)
    if override is not None:
        output = Path(job['output'])/name
        # A device exception may only reuse an already completed immutable output.
        assert sha(output/'MANIFEST.json') == override['manifest_sha256']
        assert sha(output/'COMPLETE.json') == override['complete_sha256']
        device = override['requested_device']
    return device


def validate_runtime_device(manifest, device, expected_uuid):
    runtime = manifest.get('runtime', manifest.get('bindings', {}).get('runtime'))
    assert runtime is not None, 'Missing actual runtime evidence'
    assert runtime['requested_device'] == device
    assert runtime['CUDA_VISIBLE_DEVICES'] == device
    actual_uuid = runtime['gpu']['uuid'].removeprefix('GPU-')
    assert actual_uuid == expected_uuid.removeprefix('GPU-'), 'Actual physical GPU differs from assigned device'


def execute_step(plan, job, name, script, request, gpu, run_id, frozen_runner_sha):
    if sha(__file__) != frozen_runner_sha:
        raise ValueError('Orchestration source changed')
    for source, digest in plan['source_sha256'].items():
        if sha(Path(plan['evaluation_tools'])/source) != digest:
            raise ValueError('Evaluation source changed: '+source)
    root = Path(plan['output_root'])
    request_path = root/'requests'/job['endpoint_id']/(name+'.json')
    immutable(request_path, request)
    output = Path(job['output'])/name
    attempt = root/'attempts'/run_id/job['endpoint_id']/name
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(plan['evaluation_tools'])/script), '--request', str(request_path), '--output', str(output)]
    device = step_device(plan, job, name) if gpu else ''
    if gpu:
        command += ['--device', device]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    env['PYTHONUNBUFFERED'] = '1'
    immutable(attempt/'started.json', {'utc': utc(), 'command': command, 'request_sha256': sha(request_path),
                                      'lane': job['lane'], 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'],
                                      'output_was_complete': (output/'COMPLETE.json').exists()})
    print(json.dumps({'utc': utc(), 'event': 'started', 'endpoint': job['endpoint_id'], 'step': name, 'lane': job['lane']}), flush=True)
    with (attempt/'process.log').open('xb') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    immutable(attempt/'finished.json', {'utc': utc(), 'returncode': result.returncode,
                                       'process_log_sha256': sha(attempt/'process.log')})
    if result.returncode != 0:
        raise RuntimeError(f'{job["endpoint_id"]}/{name} exited{result.returncode}; inspect {attempt}/process.log')
    if not (output/'COMPLETE.json').is_file():
        raise RuntimeError('Successful process did not produce a completion marker: '+str(output))
    if gpu:
        uuid_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        uuids = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in uuid_rows.strip().splitlines()}
        validate_runtime_device(read(output/'MANIFEST.json'), device, uuids[device])
    print(json.dumps({'utc': utc(), 'event': 'complete', 'endpoint': job['endpoint_id'], 'step': name, 'lane': job['lane']}), flush=True)
    return {'request_sha256': sha(request_path), 'complete_sha256': sha(output/'COMPLETE.json'), 'manifest_sha256': sha(output/'MANIFEST.json')}


def execute_plan(plan, run_id, frozen_runner_sha):
    stop = threading.Event()
    def lane(number):
        completed = []
        try:
            for job in [j for j in plan['jobs'] if j['lane'] == number]:
                results = {}
                steps = [
                    ('native_bdd', 'run_native_bdd_validation.py', lambda: job['native_bdd_request'], True),
                    ('bdd_predictions', 'run_prediction_archive.py', lambda: job['prediction_requests']['bdd'], True),
                    ('bdd_common', 'score_common_archive.py', lambda: scoring_request(plan, job, 'bdd'), False),
                    ('kitti_predictions', 'run_prediction_archive.py', lambda: job['prediction_requests']['kitti'], True),
                    ('kitti_common', 'score_common_archive.py', lambda: scoring_request(plan, job, 'kitti'), False),
                    ('kitti_native', 'score_native_kitti_archive.py', lambda: scoring_request(plan, job, 'kitti', True), False)]
                for name, script, request, gpu in steps:
                    if stop.is_set():
                        return completed
                    results[name] = execute_step(plan, job, name, script, request(), gpu, run_id, frozen_runner_sha)
                immutable(Path(job['output'])/'ENDPOINT_COMPLETE.json', {'endpoint_id': job['endpoint_id'], 'outputs': results})
                completed.append(job['endpoint_id'])
            return completed
        except BaseException:
            stop.set()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, i) for i in range(2)]
        results = []
        errors = []
        for future in futures:
            try:
                results.extend(future.result())
            except BaseException:
                errors.append(traceback.format_exc())
        if errors:
            raise RuntimeError('\n'.join(errors))
    assert len(results) == len(set(results)) == 48
    return {j['endpoint_id']: sha(Path(j['output'])/'ENDPOINT_COMPLETE.json') for j in plan['jobs']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--recovery-sha256')
    a = parser.parse_args()
    assert sha(a.plan) == a.plan_sha256
    plan = read(a.plan)
    if a.recovery is not None:
        assert sha(a.recovery) == a.recovery_sha256
        recovery = read(a.recovery)
        assert recovery['plan_sha256'] == a.plan_sha256
        plan['_recovery'] = recovery
    if a.check_only:
        print(json.dumps(validate_plan(plan), indent=2))
        return
    root = Path(plan['output_root'])
    root.mkdir(exist_ok=True)
    with (root/'coordinator.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+uuid.uuid4().hex[:8]
        metadata = root/'launches'/run_id
        metadata.mkdir(parents=True)
        runner_sha = sha(__file__)
        immutable(metadata/'launch.json', {'utc': utc(), 'pid': os.getpid(), 'plan_sha256': a.plan_sha256,
                                           'runner_sha256': runner_sha, 'run_id': run_id, 'recovery_sha256': a.recovery_sha256})
        try:
            immutable(metadata/'preflight.json', validate_plan(plan))
            outputs = execute_plan(plan, run_id, runner_sha)
            immutable(root/'EVALUATION_COMPLETE.json', {'schema': 'fixed_evaluation_complete_v1',
                                                       'plan_sha256': a.plan_sha256, 'endpoint_complete_sha256': outputs})
            immutable(metadata/'finished.json', {'status': 'complete', 'utc': utc()})
        except BaseException:
            immutable(metadata/'failure.json', {'status': 'failed', 'utc': utc(), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
