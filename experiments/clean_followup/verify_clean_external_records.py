"""Recalculate every fixed external outcome directly from native precision curves."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='work/evidence/clean_followup')
    args = parser.parse_args()
    root = Path(args.root)
    external = root / 'external_kitti'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = load(external / 'frozen_models.json')
    assert len(frozen['models']) == 25
    assert len({r['name'] for r in frozen['models']}) == 25
    assert frozen['input_audit_sha256'] == sha(external / 'input_audit.json')
    file_audit = load(root / 'external_artifact_file_verification.json')
    assert file_audit['status'] == 'passed' and file_audit['models'] == 25
    audited = {row['name']: row for row in file_audit['runs']}
    transfer = load(root / 'evaluation_record_transfer_manifest.json') if (root / 'evaluation_record_transfer_manifest.json').exists() else []
    for item in transfer:
        path = root / item['path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    records, rows = {}, []
    for model in frozen['models']:
        folder = external / 'results' / model['name']
        metrics = load(folder / 'metrics.json')
        config = load(folder / 'config.json')
        assert metrics['status'] == 'complete' and metrics['images'] == 7481
        assert metrics['name'] == model['name'] and metrics['group'] == model['group']
        assert metrics['checkpoint_sha256'] == model['checkpoint_sha256']
        assert config['model'] == model and config['settings'] == frozen['inference']
        assert config['frozen_manifest_sha256'] == sha(external / 'frozen_models.json')
        assert metrics['config_sha256'] == sha(folder / 'config.json')
        assert audited[model['name']]['images'] == 7481 and audited[model['name']]['verified_batch_markers'] == 936
        assert audited[model['name']]['predictions_sha256'] == metrics['predictions_sha256']
        row = dict(name=model['name'], group=model['group'])
        if 'arm' in model:
            row.update(arm=model['arm'], seed=model['seed'])
        for category in ['car', 'pedestrian']:
            curves = [[float(x) for x in line.split()] for line in (folder / f'stats_{category}_detection.txt').read_text().splitlines() if line.strip()]
            assert len(curves) == 3 and all(len(c) == 41 for c in curves)
            for index, difficulty in enumerate(['easy', 'moderate', 'hard']):
                curve = curves[index]
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in curve)
                assert all(a >= b - 1e-14 for a, b in zip(curve, curve[1:])), model['name']
                ap = statistics.mean(curve[1:])
                assert abs(ap - metrics['class_AP_R40'][category][index]) < 1e-12
                row[f'{category}_{difficulty}_AP_R40'] = ap
        row['two_class_moderate_AP_R40'] = statistics.mean([row['car_moderate_AP_R40'], row['pedestrian_moderate_AP_R40']])
        assert abs(row['two_class_moderate_AP_R40'] - metrics['primary_mean_Car_Pedestrian_moderate_AP_R40']) < 1e-12
        row['checkpoint_sha256'] = model['checkpoint_sha256']
        row['predictions_sha256'] = metrics['predictions_sha256']
        rows.append(row)
        records[model['name']] = row
    summaries = {}
    for prefix in ['earlier', 'clean']:
        summaries[prefix] = {}
        for arm in ['control', 'mse', 'cwd']:
            group = [records[f'{prefix}_{arm}_seed{s}_e20'] for s in range(3)]
            summaries[prefix][arm] = {}
            for key in ['car_moderate_AP_R40', 'pedestrian_moderate_AP_R40', 'two_class_moderate_AP_R40']:
                values = [r[key] for r in group]
                differences = [100 * (r[key] - records[f'{prefix}_control_seed{s}_e20'][key]) for s, r in enumerate(group)]
                summaries[prefix][arm][key] = dict(values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                                                  paired_delta_pp=differences, mean_delta_pp=statistics.mean(differences), delta_sample_sd_pp=statistics.stdev(differences))
    result = dict(status='passed', fixed_models=25, images_each=7481, precision_curves_verified=150,
                  aggregation='Mean of native precision positions 1 through 40; position 0 excluded.',
                  predictions_independently_rehashed_on_server=True,
                  prediction_file_audit_sha256=sha(root / 'external_artifact_file_verification.json'),
                  summary=summaries, runs=rows,
                  limitations=['External public labeled KITTI training partition, not an official hidden-test score.',
                               'Car/Pedestrian only, with native ignores and difficulty filters; different from BDD ten-class AP.',
                               'No tuning or selection by these outcomes; three seeds share one starting model per phase.'])
    (root / 'external_record_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'external_all_models.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status='passed', models=25, summaries=summaries)))


if __name__ == '__main__':
    main()
