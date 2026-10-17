"""Read-only comparison of recorded initial parameters and sampled inputs.

Run on partial or completed base records. This checks the two recorded batches
per epoch, not every batch, and does not claim independent COCO pretraining.
"""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    records = root / 'base_records'
    comparisons, pending = [], []
    for model in protocol['base']['models']:
        for left, right in itertools.combinations(protocol['pair_seeds'], 2):
            dirs = [records / f'{model}_seed{s}' for s in (left, right)]
            names = [d.name for d in dirs]
            required = ['config.json', 'initial_parameters.json']
            if not all((d / f).exists() for d in dirs for f in required):
                pending.append(names)
                continue
            configs = [json.loads((d / 'config.json').read_text()) for d in dirs]
            for config, seed in zip(configs, (left, right)):
                assert config['seed'] == seed and config['model'] == model
                assert config['protocol_sha256'] == digest(root / 'protocol.json')
                assert config['source_sha256'] == protocol['base'][model + '_source_sha256']
            initials = [json.loads((d / 'initial_parameters.json').read_text()) for d in dirs]
            assert initials[0].keys() == initials[1].keys(), names
            different = []
            for key in initials[0]:
                assert initials[0][key]['shape'] == initials[1][key]['shape'], (names, key)
                if initials[0][key]['sha256'] != initials[1][key]['sha256']:
                    different.append(key)
            assert different, (names, 'identical initial parameter records')
            assert not any('dfl.conv.weight' in key for key in different), names
            epochs = [{int(p.stem.split('_')[1]): p for p in d.glob('epoch_*.json')} for d in dirs]
            common = sorted(epochs[0].keys() & epochs[1].keys())
            evidence = []
            for epoch in common:
                paths = [e[epoch] for e in epochs]
                inputs = [json.loads(p.read_text())['first_two_input_sha256'] for p in paths]
                assert len(inputs[0]) == len(inputs[1]) == 2, (names, epoch)
                distinct = [a != b for a, b in zip(*inputs)]
                assert all(distinct), (names, epoch, 'matching sampled actual input')
                evidence.append({'epoch': epoch, 'first_two_input_sha256': inputs,
                                 'epoch_record_sha256': [digest(p) for p in paths]})
            comparisons.append({'runs': names, 'initial_record_sha256': [digest(d / 'initial_parameters.json') for d in dirs],
                                'different_initial_parameter_names': different,
                                'same_official_COCO_initial_source': True,
                                'common_recorded_epochs': len(common),
                                'both_sampled_batches_differ_at_each_common_epoch': True if common else None,
                                'input_evidence': evidence})
    return {'status': 'passed_for_available_records', 'utc': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': digest(root / 'protocol.json'),
            'audit_script_sha256': digest(Path(__file__)), 'comparisons': comparisons,
            'pending_run_pairs': pending,
            'scope': 'Initial parameter hashes and first two actual minibatch hashes per common completed epoch only; remaining seeds/epochs pending. Fixed COCO pretraining is shared within each architecture.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))
