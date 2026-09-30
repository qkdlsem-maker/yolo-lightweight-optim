"""Independent, read-only recalculation of the completed clean training evidence."""
import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='work/evidence/clean_followup')
    parser.add_argument('--verify-transfer', action='store_true', help='Also check the private local transfer manifest when available.')
    args = parser.parse_args()
    root = Path(args.root)
    load = lambda path: json.loads((root / path).read_text(encoding='utf-8'))
    transfer = load('training_evidence_transfer_manifest.json') if args.verify_transfer else []
    for item in transfer:
        payload = (root / item['path']).read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256'], item['path']
    audit = load('numerical_state_audit.json')
    assert audit['status'] == 'passed' and not audit['includes_active_runs']
    reported = load('confirmatory_summary.json')
    frozen = {r['name']: r for r in load('external_kitti/frozen_models.json')['models']}
    arms = ['control', 'mse', 'cwd']
    records, rows, epoch_hashes = {}, [], {}
    for arm, seed in itertools.product(arms, range(3)):
        name = f'{arm}_seed{seed}_e20'
        folder = f'confirmatory/{name}'
        config = load(folder + '/config.json')
        expected = dict(arm=arm, seed=seed, epochs=20, batch=64, workers=4,
                        train_images=69863, lr=.0001, kd_weight=.3, temperature=4.,
                        student_sha256='63c03589fe1dcd7a5d5d2996fa64d51be250b3fdbfb09163b9d93e1882fe4273',
                        teacher_sha256='542ffff61f95ae5ebda53b1279cf9b238e754fe9b149b0c98fc686d55e9a7bd2',
                        script_sha256='f3baa82158110bbbc46bcc9053f78101d3fd8e0c18ebd4e5751d5dbf7d96cae1')
        for key, value in expected.items():
            assert config[key] == value, (name, key)
        delta = load(folder + '/parameter_delta.json')
        assert delta['changed_tensors'] == 183 and delta['fixed_dfl_preserved']
        grad = load(folder + '/gradient_check.json')
        assert grad['student_nonzero_grad_tensors'] == 183
        if arm != 'control':
            assert grad['adapter_grad_tensors'] == 6
        metrics = load(folder + '/COMPLETE.json')
        assert metrics == load(folder + '/ema_epoch20_metrics.json')
        assert frozen['clean_' + name]['checkpoint_sha256'] == metrics['checkpoint_sha256']
        for metric, per_class in [('map50', 'ap50'), ('map50_95', 'ap50_95')]:
            assert len(metrics[per_class]) == 10
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in metrics[per_class])
            assert abs(statistics.mean(metrics[per_class]) - metrics[metric]) < 1e-12
        epochs = [load(folder + f'/epoch_{e:02d}.json') for e in range(1, 21)]
        assert len(list((root / folder).glob('epoch_*.json'))) == 20
        for epoch, entry in enumerate(epochs, 1):
            assert entry['epoch'] == epoch and entry['batches'] == 1092
            assert math.isfinite(entry['det_loss']) and math.isfinite(entry['feature_loss'])
            assert abs(entry['lr'] - (.00001 + .00009 * (1 + math.cos(math.pi * (epoch - 1) / 19)) / 2)) < 1e-12
            hashes = entry['first_two_input_sha256']
            assert len(hashes) == 2 and all(len(h) == 64 for h in hashes)
            epoch_hashes[arm, seed, epoch] = tuple(hashes)
            if not math.isfinite(entry['median_unclipped_grad_norm']):
                assert entry['optimizer_steps_skipped'] > 0
        skips = sum(e['optimizer_steps_skipped'] for e in epochs)
        numerics = audit['runs'][name]
        assert numerics['complete'] and numerics['snapshot_epoch'] == 20
        assert numerics['attempted_minibatches'] == 21840
        assert numerics['skipped_optimizer_steps'] == skips
        assert numerics['applied_optimizer_steps'] == numerics['EMA_updates'] == 21840 - skips
        assert not any(numerics['nonfinite_tensor_counts'].values())
        records[arm, seed] = metrics
        rows.append(dict(arm=arm, seed=seed, AP50=metrics['map50'], AP50_95=metrics['map50_95'],
                         attempted_minibatches=21840, skipped_updates=skips, applied_updates=21840-skips,
                         checkpoint_sha256=metrics['checkpoint_sha256'], state_sha256=metrics['state_sha256']))
    within = between = 0
    for epoch in range(1, 21):
        for seed in range(3):
            for a, b in itertools.combinations(arms, 2):
                assert epoch_hashes[a, seed, epoch] == epoch_hashes[b, seed, epoch]
                within += 1
        for s, t in itertools.combinations(range(3), 2):
            for a, b in itertools.product(arms, repeat=2):
                assert epoch_hashes[a, s, epoch] != epoch_hashes[b, t, epoch]
                between += 1
    assert len({records['control', s]['state_sha256'] for s in range(3)}) == 3
    summaries = {}
    for arm in arms:
        group = [r for r in rows if r['arm'] == arm]
        for row in group:
            for metric, key in [('AP50', 'map50'), ('AP50_95', 'map50_95')]:
                row[f'delta_{metric}_pp'] = 100 * (row[metric] - records['control', row['seed']][key])
        summaries[arm] = {}
        for key in ['AP50', 'AP50_95', 'delta_AP50_pp', 'delta_AP50_95_pp']:
            values = [r[key] for r in group]
            summaries[arm][key] = dict(values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
            for quantity in ['mean', 'sample_sd']:
                assert abs(summaries[arm][key][quantity] - reported['summary'][arm][key][quantity]) < 1e-12
    result = dict(status='passed', transferred_files_verified=len(transfer), within_seed_epoch_pairs=within,
                  between_seed_epoch_pairs=between, independent_baseline_initializations=1,
                  primary_evaluation_batch=32, runs=rows, summary=summaries,
                  limitations=['First two actual minibatches per epoch are sampled hashes, not an all-batch equality proof.',
                               'Three data/augmentation seeds share one trained starting student and teacher.',
                               'AMP skips and raw nonfinite gradient diagnostics are preserved; final states are finite.',
                               'Descriptive conditional effects; no equivalence or universal objective ranking.'])
    (root / 'independent_training_record_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], within=within, between=between,
                         means={a: summaries[a]['AP50_95']['mean'] for a in arms},
                         delta_pp={a: summaries[a]['delta_AP50_95_pp']['mean'] for a in arms})))


if __name__ == '__main__':
    main()
