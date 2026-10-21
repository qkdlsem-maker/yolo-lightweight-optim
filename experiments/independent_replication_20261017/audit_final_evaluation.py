"""Read-only numeric replay from completed outputs; never loads a model.

Uses the pinned metric implementations, not an independent metric definition.
Replays native BDD from saved TP/confidence/class statistics, common AP from
saved prediction boxes, and native KITTI AP_R40 from retained precision curves.
No inference, training, C++ reevaluation, score tuning, or original-data writes.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
import yaml
from ultralytics.utils.metrics import DetMetrics
from common_ap_core import evaluate
from prediction_archive import PredictionArchive, canonical, digest, sha256_file
from prediction_io import common_rows, kitti_text, image_metadata, INFERENCE
from run_native_bdd_validation import verify_complete, native_hashes
from score_common_archive import verify_output as verify_common
from score_native_kitti_archive import verify_output as verify_native_kitti, parse_curves, label_snapshot, CORE


def read(path):
    return json.loads(Path(path).read_bytes())


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
        f.flush()
        os.fsync(f.fileno())


def trace_check(traces, count):
    expected = [[min(32, count-i), 3, 640, 640] for i in range(0, count, 32)]
    assert [t['shape'] for t in traces] == expected
    assert all(t['dtype'] == 'torch.float32' and t['device'] == 'cuda:0' for t in traces)


def native_bdd(folder, request):
    manifest = read(folder/'MANIFEST.json')
    assert manifest['request'] == request
    assert manifest['native_validation_source_sha256'] == native_hashes()
    assert not verify_complete(folder, manifest)['new_inference']
    score = read(folder/'score.json')
    audit = read(folder/'input_audit.json')
    images = read(request['image_manifest']['path'])
    assert audit['images'] == [im['id'] for im in images] == list(range(1, 10001))
    assert audit['native_loaded_labels_match_raw_float32_unique_rows']
    assert not audit['training_context'] and audit['workers'] == 4
    trace_check(audit['actual_forward_inputs'], len(images))
    files = sorted((folder/'native_stats').glob('batch_*.npz'))
    assert len(files) == len(audit['batches']) == math.ceil(len(images)/32)
    arrays = {}
    all_ids = []
    for i, (file, batch) in enumerate(zip(files, audit['batches'])):
        assert file.name == f'batch_{i:04d}.npz' and batch['batch_index'] == i
        ledger = batch['images']
        ids = [x['image_id'] for x in ledger]
        assert ids == [im['id'] for im in images[i*32:(i+1)*32]]
        all_ids.extend(ids)
        with np.load(file, allow_pickle=False) as record:
            values = {k: record[k] for k in record.files}
        assert set(values) == {'tp', 'conf', 'pred_cls', 'target_cls', 'target_img', 'prediction_image_id', 'target_image_id'}
        assert all(np.isfinite(v).all() for v in values.values())
        assert values['tp'].shape == (len(values['conf']), 10)
        assert set(np.unique(values['tp'])) <= {0, 1}
        assert ((values['conf'] >= .001) & (values['conf'] <= 1)).all()
        for k in ['pred_cls', 'target_cls']:
            assert ((values[k] >= 0) & (values[k] < 10) & (values[k] == np.floor(values[k]))).all()
        assert values['prediction_image_id'].tolist() == [x['image_id'] for x in ledger for _ in range(x['predictions'])]
        assert values['target_image_id'].tolist() == [x['image_id'] for x in ledger for _ in range(x['targets'])]
        for k, value in values.items():
            arrays.setdefault(k, []).append(value)
    assert all_ids == audit['images']
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    assert len(arrays['target_cls']) == audit['native_loaded_boxes'] == score['native_boxes']
    assert score['images'] == 10000 and score['inference'] == INFERENCE
    names = yaml.safe_load(Path(request['data_yaml']['path']).read_bytes())['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    replay = DetMetrics(names=names)
    if arrays['tp'].any():
        replay.process(**{k: arrays[k] for k in ['tp', 'conf', 'pred_cls', 'target_cls']})
    assert replay.results_dict == score['results'], 'Native BDD statistics do not reproduce results'
    assert np.array_equal(np.asarray(replay.box.all_ap), np.asarray(score['class_AP_by_IoU']))
    assert np.array_equal(np.asarray(replay.box.ap_class_index), np.asarray(score['AP_class_indices']))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in score['results'].values())
    return {'images': len(images), 'raw_label_rows': audit['raw_label_rows'],
            'native_boxes': score['native_boxes'], 'prediction_rows': len(arrays['conf']),
            'saved_statistics_exactly_reproduce_all_results_and_class_AP': True,
            'actual_batches_FP32_640_square_and_image_ledger_verified': True,
            'results': score['results'], 'score_sha256': sha256_file(folder/'score.json')}


def archive_and_common(root, request, dataset, plan, endpoint):
    folder = root/(dataset+'_predictions')
    archived = read(folder/'MANIFEST.json')
    bindings = archived['bindings']
    assert bindings['endpoint_id'] == endpoint and bindings['dataset_id'] == dataset
    assert bindings['checkpoint_sha256'] == request['checkpoint']['sha256']
    assert bindings['protocol_sha256'] == request['protocol']['sha256']
    assert bindings['inference'] == INFERENCE
    assert bindings['source_sha256'] == {**request['source_sha256'], 'frozen_request': digest(canonical(request))}
    images = read(request['image_manifest']['path'])
    assert archived['images'] == images
    with PredictionArchive(folder, bindings, images) as archive:
        verified = archive.verify()
        assert verified['complete']
        traces = []
        for index in range(archive.batch_count):
            traces.extend(read(folder/'batches'/f'{index:06d}'/'COMMIT.json')['actual_forward_inputs'])
        trace_check(traces, len(images))
        predictions = []
        count = 0
        native_counts = {'mapped': 0, 'empty': 0, 'zero_area': 0}
        for record in archive.iter_records():
            count += len(record['detections'])
            predictions.extend(common_rows(record))
            if dataset == 'kitti':
                text = kitti_text(record).encode('ascii')
                name = Path(record['image']['file_name']).stem+'.txt'
                assert (root/'kitti_native/predictions'/name).read_bytes() == text
                native_counts['mapped'] += text.count(b'\n')
                native_counts['empty'] += not text
                native_counts['zero_area'] += sum((x2 == x1 or y2 == y1) and c in (0, 2)
                                                   for x1,y1,x2,y2,score,c in record['detections'])
        projection_sha = digest(canonical(predictions))
        output = root/(dataset+'_common')
        manifest = read(output/'MANIFEST.json')
        assert not verify_common(output, manifest)['new_scores_computed']
        req = manifest['request']
        assert req['dataset'] == dataset and Path(req['archive_path']) == folder
        assert req['archive_manifest_sha256'] == sha256_file(folder/'MANIFEST.json')
        assert req['archive_complete_sha256'] == sha256_file(folder/'COMPLETE.json')
        assert req['source_sha256'] == plan['common_source_sha256']
        variants = ['bdd_common'] if dataset == 'bdd' else ['kitti_source_ignore', 'kitti_no_ignore_sensitivity']
        assert req['annotation_views'] == {v: plan['annotation_views'][v] for v in variants}
        rows = {}
        for variant in variants:
            ref = req['annotation_views'][variant]
            assert sha256_file(ref['path']) == ref['sha256']
            annotation = read(ref['path'])
            assert [image_metadata(x) for x in annotation['images']] == [image_metadata(x) for x in images]
            saved = read(output/(variant+'.json'))
            assert saved['shared_common_prediction_projection_sha256'] == projection_sha
            assert saved['all_class_predictions_preserved'] == count
            assert saved['mapped_person_car_predictions'] == len(predictions)
            replay = evaluate(annotation['images'], annotation['annotations'], predictions, include_matches=False)
            assert replay == saved['score'], 'Common AP replay differs: '+variant
            rows[variant] = {'all_numeric_fields_replayed_exactly': True, 'score_sha256': sha256_file(output/(variant+'.json')),
                             'mean_AP50_95': saved['score']['mean_AP50_95'], 'class_AP50_95': saved['score']['class_AP50_95']}
        # Read-only replay must leave all archived prediction bytes untouched.
        assert archive.verify() == verified
    return {'images': len(images), 'all_class_predictions': count, 'mapped_predictions': len(predictions),
            'projection_sha256': projection_sha, 'scores': rows, 'native_export_counts': native_counts}


def native_kitti(folder, plan, common):
    manifest = read(folder/'MANIFEST.json')
    assert not verify_native_kitti(folder, manifest)['new_scores_computed']
    request = manifest['request']
    assert request['source_sha256'] == plan['kitti_source_sha256']
    assert request['native_core_paths'] == plan['native_core_paths'] and request['label_zip'] == plan['label_zip']
    for name, expected in CORE.items():
        assert sha256_file(plan['native_core_paths'][name]) == expected
    archive = Path(request['archive_path'])
    assert archive == folder.parent/'kitti_predictions'
    assert request['archive_manifest_sha256'] == sha256_file(archive/'MANIFEST.json')
    assert request['archive_complete_sha256'] == sha256_file(archive/'COMPLETE.json')
    images = read(archive/'MANIFEST.json')['images']
    labels = label_snapshot(plan['label_zip']['path'], images)
    assert len(labels) == len(list((folder/'ground_truth').glob('*.txt'))) == len(list((folder/'predictions').glob('*.txt'))) == 7481
    for name, raw in labels.items():
        assert (folder/'ground_truth'/name).read_bytes() == raw
    curves = parse_curves(folder/'curves')
    score = read(folder/'score.json')
    assert curves == score['classes']
    mean = math.fsum(curves[c]['moderate']['AP_R40'] for c in ['car', 'pedestrian'])/2
    assert mean == score['mean_moderate_AP_R40']
    assert score['images'] == 7481 and score['all_class_predictions_preserved'] == common['all_class_predictions']
    for field, key in [('mapped_predictions_exported','mapped'), ('empty_prediction_files','empty'), ('zero_area_mapped_predictions_exported','zero_area')]:
        assert score[field] == common['native_export_counts'][key]
    return {'images': 7481, 'exported_text_exactly_matches_prediction_archive': True,
            'ground_truth_exactly_matches_fixed_zip': True, 'AP_R40_reconstructed_exactly_from_saved_curves': True,
            'mean_moderate_AP_R40': mean, 'class_AP_R40': {c: {d: v['AP_R40'] for d,v in rows.items()} for c,rows in curves.items()},
            'score_sha256': sha256_file(folder/'score.json'),
            'limitation': 'Native C++ matching was not rerun; audit checks pinned core provenance, exact text exports, labels and retained precision integration.'}


def audit(plan, job, plan_sha):
    root = Path(job['output'])
    before = read(root/'ENDPOINT_COMPLETE.json')
    before_sha = sha256_file(root/'ENDPOINT_COMPLETE.json')
    assert before['endpoint_id'] == job['endpoint_id']
    expected = ['native_bdd','bdd_predictions','bdd_common','kitti_predictions','kitti_common','kitti_native']
    assert set(before['outputs']) == set(expected)
    for name, item in before['outputs'].items():
        assert item['manifest_sha256'] == sha256_file(root/name/'MANIFEST.json')
        assert item['complete_sha256'] == sha256_file(root/name/'COMPLETE.json')
        request_path = Path(plan['output_root'])/'requests'/job['endpoint_id']/(name+'.json')
        assert item['request_sha256'] == sha256_file(request_path)
    for name, digest_value in plan['source_sha256'].items():
        assert sha256_file(Path(plan['evaluation_tools'])/name) == digest_value
    refs = [plan['roster'], plan['protocol'], plan['label_zip']]
    for spec in [job['native_bdd_request'], *job['prediction_requests'].values()]:
        refs += [spec[n] for n in ['checkpoint','protocol','image_manifest','dataset_manifest','data_yaml'] if n in spec]
    for ref in refs:
        assert sha256_file(ref['path']) == ref['sha256']
    nb = native_bdd(root/'native_bdd', job['native_bdd_request'])
    bdd = archive_and_common(root, job['prediction_requests']['bdd'], 'bdd', plan, job['endpoint_id'])
    kitti = archive_and_common(root, job['prediction_requests']['kitti'], 'kitti', plan, job['endpoint_id'])
    nk = native_kitti(root/'kitti_native', plan, kitti)
    assert sha256_file(root/'ENDPOINT_COMPLETE.json') == before_sha
    for name, item in before['outputs'].items():
        assert item['manifest_sha256'] == sha256_file(root/name/'MANIFEST.json')
        assert item['complete_sha256'] == sha256_file(root/name/'COMPLETE.json')
    # Repeat integrity inventory after replay without redoing metric computation.
    verify_complete(root/'native_bdd', read(root/'native_bdd/MANIFEST.json'))
    for dataset in ['bdd','kitti']:
        verify_common(root/(dataset+'_common'), read(root/(dataset+'_common')/'MANIFEST.json'))
    verify_native_kitti(root/'kitti_native', read(root/'kitti_native/MANIFEST.json'))
    return {'status': 'passed', 'utc': utc(), 'endpoint_id': job['endpoint_id'],
            'plan_sha256': plan_sha, 'endpoint_complete_sha256': before_sha,
            'audit_source_sha256': sha256_file(__file__), 'native_bdd': nb, 'bdd_common': bdd,
            'kitti_common': kitti, 'native_kitti': nk,
            'scope': 'Independent read-only artifact and numerical replay using the same pinned metric implementations. No model loading, training, inference, C++ reevaluation or output overwrite.',
            'limitations': ['This validates saved computations, not a new untouched evaluation population or independent metric implementation.',
                            'Does not reconstruct native BDD IoU matches from image pixels; saved per-detection TP statistics are replayed.',
                            'Native KITTI matching core is provenance-bound; only saved precision integration is recomputed.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert sha256_file(args.plan) == args.plan_sha256
    plan = read(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'audit.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [j for j in plan['jobs'] if args.endpoint is None or j['endpoint_id'] == args.endpoint]
        assert len(jobs) == (48 if args.endpoint is None else 1)
        for job in jobs:
            target = args.output/(job['endpoint_id']+'.json')
            marker = Path(job['output'])/'ENDPOINT_COMPLETE.json'
            if target.exists():
                prior = read(target)
                assert prior['status'] == 'passed' and prior['plan_sha256'] == args.plan_sha256
                assert prior['audit_source_sha256'] == sha256_file(__file__)
                assert prior['endpoint_complete_sha256'] == sha256_file(marker)
                continue
            while not marker.exists():
                if not args.watch:
                    break
                # Stop if the evaluation coordinator reports a technical failure.
                assert not list((Path(plan['output_root'])/'launches').glob('*/failure.json')), 'Evaluation failure requires review'
                time.sleep(30)
            if not marker.exists():
                continue
            print(json.dumps({'utc':utc(),'event':'audit_started','endpoint':job['endpoint_id']}),flush=True)
            try:
                result = audit(plan, job, args.plan_sha256)
                new(target, result)
            except BaseException:
                new(args.output/(job['endpoint_id']+'_failure_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'),
                    {'utc':utc(),'endpoint':job['endpoint_id'],'traceback':traceback.format_exc(),'audit_source_sha256':sha256_file(__file__)})
                raise
            print(json.dumps({'utc':utc(),'event':'audit_passed','endpoint':job['endpoint_id']}),flush=True)
        if args.endpoint is None and all((args.output/(j['endpoint_id']+'.json')).exists() for j in jobs):
            result = {'status':'passed','plan_sha256':args.plan_sha256,
                      'audit_source_sha256':sha256_file(__file__),
                      'proof_sha256':{j['endpoint_id']:sha256_file(args.output/(j['endpoint_id']+'.json')) for j in jobs}}
            if (args.output/'AUDITS_COMPLETE.json').exists():
                assert read(args.output/'AUDITS_COMPLETE.json') == result
            else:
                new(args.output/'AUDITS_COMPLETE.json', result)


if __name__ == '__main__':
    main()
