"""Derive descriptive paired summaries from all nine completed runs; no retraining."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(values):
    return {'values': values, 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values)}


def main(root):
    primary = json.loads((root / 'confirmatory_summary.json').read_text())
    audit = json.loads((root / 'numerical_state_audit.json').read_text())
    assert primary['status'] == 'complete' and primary['all_input_pairing_checks_passed']
    assert len(primary['runs']) == len(audit['runs']) == 9 and audit['status'] == 'passed'
    metrics = {p.stem: json.loads(p.read_text()) for p in (root / 'stratified').glob('*.json')}
    assert len(metrics) == 16
    assert all(x['statistics_reproduce_validator_AP'] for x in metrics.values())
    groups = ['all10', 'daytime', 'night', 'dawn/dusk']
    rows = []
    aggregate = {}
    for arm in ['control', 'mse', 'cwd']:
        aggregate[arm] = {}
        for group in groups:
            aggregate[arm][group] = {}
            for metric in ['AP50', 'AP50_95']:
                values, deltas = [], []
                for seed in range(3):
                    x = metrics[f'{arm}_seed{seed}_e20']
                    c = metrics[f'control_seed{seed}_e20']
                    if group == 'all10':
                        v = x['overall'][metric + '_present_classes']
                        base = c['overall'][metric + '_present_classes']
                    else:
                        v = x['timeofday'][group]['road9_' + metric]
                        base = c['timeofday'][group]['road9_' + metric]
                    values.append(v)
                    deltas.append(100 * (v - base))
                    rows.append([arm, seed, group, metric, v, base, deltas[-1]])
                aggregate[arm][group][metric] = summarize(values)
                aggregate[arm][group]['delta_' + metric + '_pp'] = summarize(deltas)
    with (root / 'final_secondary_paired.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arm', 'seed', 'group', 'metric', 'value', 'matched_control', 'delta_pp'])
        writer.writerows(rows)
    counts = []
    for name, run in sorted(audit['runs'].items()):
        assert run['complete'] and run['snapshot_epoch'] == 20
        assert sum(run['nonfinite_tensor_counts'].values()) == 0
        assert run['all_recorded_objective_means_finite']
        assert run['executed_script_hash_matches_config']
        assert run['attempted_minibatches'] - run['skipped_optimizer_steps'] == run['EMA_updates'] == run['applied_optimizer_steps']
        counts.append([name, run['attempted_minibatches'], run['skipped_optimizer_steps'], run['applied_optimizer_steps']])
    with (root / 'final_optimizer_updates.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'attempted_minibatches', 'AMP_skips', 'applied_optimizer_steps'])
        writer.writerows(counts)
    sensitivity = json.loads((root / 'annotation_sensitivity.json').read_text())['models']
    assert len(sensitivity) == 16
    report = {
        'primary': primary['summary'], 'secondary': aggregate,
        'completed_training_runs': 9, 'completed_secondary_evaluations': 16,
        'optimizer_updates': counts,
        'annotation_AP50_delta_pp_range': [min(x['delta_AP50_pp'] for x in sensitivity.values()), max(x['delta_AP50_pp'] for x in sensitivity.values())],
        'interpretation': 'Descriptive paired comparison across three data/augmentation seeds, conditional on one common trained baseline. No consistent positive primary-endpoint benefit; no equivalence or general superiority claim.'
    }
    (root / 'final_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed_training_runs': 9, 'completed_secondary_evaluations': 16,
                      'secondary_AP95_deltas_pp': {a: {g: aggregate[a][g]['delta_AP50_95_pp'] for g in groups} for a in ['mse','cwd']},
                      'annotation_range_pp': report['annotation_AP50_delta_pp_range']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence_directory', type=Path)
    main(parser.parse_args().evidence_directory)
