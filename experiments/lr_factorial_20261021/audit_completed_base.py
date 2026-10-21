"""Read-only CPU audit of completed new bases; never train, infer, or repair data.

Individual proofs may be reused only while every bound byte remains unchanged.
The aggregate gate requires all six proofs and all 100 sampled-input epochs.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

PROTOCOL_SHA256 = '8dac98e36fee058726b9475487896c49f53f246e960cdfc0ba7e6acb895cdc9d'
MODELS = ('nano', 'large')
SEEDS = (404, 505, 606)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class Bindings:
    def __init__(self):
        self.files = {}

    def register(self, path, digest):
        path = str(Path(path).resolve())
        if path in self.files:
            assert self.files[path] == digest, ('changed during read', path)
        self.files[path] = digest
        return digest

    def bind(self, path):
        return self.register(path, sha(path))

    def bytes(self, path):
        raw = Path(path).read_bytes()
        self.register(path, hashlib.sha256(raw).hexdigest())
        return raw

    def read(self, path):
        return json.loads(self.bytes(path))

    def load(self, path):
        return torch.load(io.BytesIO(self.bytes(path)), map_location='cpu', weights_only=False)

    def verify(self):
        for path, digest in self.files.items():
            assert sha(path) == digest, ('input mutated', path)


def verify_sources(root, bound):
    assert bound.bind(root / 'protocol.json') == PROTOCOL_SHA256
    p = bound.read(root / 'protocol.json')
    assert Path(p['root']).resolve() == root.resolve()
    assert p['pair_seeds'] == list(SEEDS)
    interpretation = bound.read(root / 'protocol_interpretation.json')
    assert interpretation['protocol_sha256'] == PROTOCOL_SHA256
    assert interpretation['canonical_design']['new_bases'] == 6
    for manifest in ('deployed_sources.json', 'recovery_sources.json', 'cache_recovery_sources.json'):
        for item in bound.read(root / manifest)['files']:
            assert bound.bind(root / item['name']) == item['sha256'], item['name']
    preflight = bound.read(root / 'BASE_PREFLIGHT_PASSED.json')
    assert preflight['status'] == 'passed' and preflight['protocol_sha256'] == PROTOCOL_SHA256
    for field, name in [('deployed_sources_sha256', 'deployed_sources.json'),
                        ('recovery_sources_sha256', 'recovery_sources.json'),
                        ('step_accounting_validation_sha256', 'step_accounting_validation.json'),
                        ('data_verification_sha256', 'data_verification.json')]:
        assert bound.bind(root / name) == preflight[field], name
    return p


def counts(value):
    assert set(value) == {'attempted', 'applied', 'skipped'}
    assert all(type(n) is int and n >= 0 for n in value.values())
    assert value['attempted'] == value['applied'] + value['skipped']
    return value


def audit_run(root, p, model_name, seed, bound):
    assert model_name in MODELS and seed in SEEDS
    name = f'{model_name}_seed{seed}'
    folder = root / 'base_records' / name
    done = bound.read(folder / 'COMPLETE.json')  # Fail before loading an active state.
    config = bound.read(folder / 'config.json')
    effective = bound.read(folder / 'effective_config.json')
    initial = bound.read(folder / 'initial_parameters.json')
    epochs = p['base']['epochs']
    assert epochs == 100
    assert done['status'] == 'complete' and done['epochs'] == epochs
    assert done['model'] == model_name and done['seed'] == seed
    assert done['finite_state'] and done['fixed_dfl_preserved'] and done['changed_parameter_tensors'] > 0
    source = Path(p['previous_root']) / 'pretrained' / ('yolov8n.pt' if model_name == 'nano' else 'yolov8l.pt')
    checkpoint = root / 'base' / name / 'weights' / 'last.pt'
    assert Path(done['checkpoint']).resolve() == checkpoint.resolve()
    assert Path(config['source']).resolve() == source.resolve()
    assert bound.bind(source) == done['source_sha256'] == p['base'][model_name + '_source_sha256']
    assert bound.bind(checkpoint) == done['checkpoint_sha256']
    assert bound.bind(folder / 'initial_parameters.json') == done['initial_parameters_sha256']
    expected = {'name': name, 'model': model_name, 'seed': seed, 'epochs': 100,
                'batch': p['base']['batch'][model_name], 'fraction': 1.0,
                'source_sha256': done['source_sha256'], 'data': p['data']['yaml'],
                'data_sha256': p['data']['yaml_sha256'], 'script_sha256': bound.bind(root / 'base_train.py'),
                'protocol_sha256': bound.bind(root / 'protocol.json')}
    for key, value in expected.items():
        assert config[key] == value, ('config', key)
    assert bound.bind(config['data']) == config['data_sha256']
    for key in ('epochs', 'nbs', 'imgsz', 'optimizer', 'lr0', 'lrf', 'cos_lr', 'momentum',
                'weight_decay', 'warmup_epochs', 'patience', 'close_mosaic', 'amp', 'workers', 'deterministic'):
        assert effective[key] == p['base'][key], ('effective config', key)
    for key in ('seed', 'data', 'batch', 'fraction'):
        assert effective[key] == config[key], ('effective config', key)
    assert effective['cache'] is False and effective['pretrained'] is True
    assert effective['save'] is True and effective['val'] is True
    rows = list(csv.DictReader(io.StringIO(bound.bytes(root / 'base' / name / 'results.csv').decode())))
    assert len(rows) == epochs and rows[-1] == done['last_row']
    assert [int(float(row[next(k for k in row if k.strip() == 'epoch')])) for row in rows] == list(range(1, epochs + 1))
    epoch_files = sorted(folder.glob('epoch_*.json'))
    assert [f.name for f in epoch_files] == [f'epoch_{i:03d}.json' for i in range(1, epochs + 1)]
    prior = {'attempted': 0, 'applied': 0, 'skipped': 0}
    inputs = []
    for number, path in enumerate(epoch_files, 1):
        record = bound.read(path)
        current = counts(record['actual_step_counts'])
        assert record['epoch'] == number and record['native_EMA_updates'] == current['attempted']
        assert all(current[k] >= prior[k] for k in prior)
        assert current['attempted'] > prior['attempted'] and current['applied'] > prior['applied']
        assert len(record['loss_items']) == 3 and all(math.isfinite(x) for x in record['loss_items'])
        digests = record['first_two_input_sha256']
        assert len(digests) == 2 and all(re.fullmatch('[0-9a-f]{64}', d) for d in digests)
        inputs.append(digests)
        prior = current
    assert counts(done['actual_step_counts']) == prior and done['native_EMA_updates'] == prior['attempted']
    state = bound.load(folder / 'resume_state.pt')
    assert state['epoch'] == epochs and counts(state['actual_step_counts']) == prior
    assert state['ema_updates'] == prior['attempted']  # Native EMA counts attempts, including AMP skips.
    for key in ('model', 'ema', 'optimizer', 'scaler', 'scheduler'):
        assert state[key] and finite(state[key]), ('nonfinite or empty saved state', key)
    for key in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
        assert key in state, ('missing recovery component', key)
    for key in ('torch_rng', 'cuda_rng'):
        assert isinstance(state[key], torch.Tensor) and state[key].dtype == torch.uint8 and state[key].numel() > 0
    released = bound.load(checkpoint)
    model = released.get('ema') if released.get('ema') is not None else released['model']
    assert model.model[-1].nc == 10 and finite(model.state_dict())
    parameter_names = set(dict(model.named_parameters()))
    assert parameter_names == set(initial)
    assert set(model.state_dict()) == set(state['ema']) == set(state['model'])
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, state['ema'][key].to(tensor.dtype)), ('checkpoint vs raw EMA', key)
    for key in parameter_names:
        assert list(state['model'][key].shape) == initial[key]['shape']
        assert state['model'][key].dtype == state['ema'][key].dtype == torch.float32
    ih = {k: v['sha256'] for k, v in initial.items()}
    rh = {k: tensor_sha(state['model'][k]) for k in parameter_names}
    eh = {k: tensor_sha(state['ema'][k]) for k in parameter_names}
    changed = [k for k in parameter_names if rh[k] != ih[k]]
    fixed = [k for k in parameter_names if 'dfl.conv.weight' in k]
    assert changed and fixed and all(rh[k] == ih[k] == eh[k] for k in fixed)
    result = {'model': model_name, 'seed': seed, 'epochs': epochs, 'actual_step_counts': prior,
              'native_EMA_updates': prior['attempted'], 'checkpoint_sha256': done['checkpoint_sha256'],
              'resume_state_sha256': bound.bind(folder / 'resume_state.pt'),
              'initial_parameter_hashes': ih, 'initial_parameter_digest': canonical_sha(ih),
              'raw_final_parameter_digest': canonical_sha(rh), 'EMA_parameter_digest': canonical_sha(eh),
              'raw_changed_parameter_tensors': len(changed), 'fixed_dfl_preserved': True,
              'checkpoint_matches_raw_EMA_at_saved_precision': True, 'finite_saved_tensors': True,
              'first_two_input_sha256_by_epoch': inputs}
    bound.verify()
    return result


def aggregate(proofs):
    rows = [proof['run'] for proof in proofs]
    assert len(rows) == 6 and {(r['model'], r['seed']) for r in rows} == set(itertools.product(MODELS, SEEDS))
    comparisons = []
    for model in MODELS:
        for a, b in itertools.combinations([r for r in rows if r['model'] == model], 2):
            for key in ('initial_parameter_digest', 'raw_final_parameter_digest', 'EMA_parameter_digest'):
                assert a[key] != b[key], (model, a['seed'], b['seed'], key)
            assert set(a['initial_parameter_hashes']) == set(b['initial_parameter_hashes'])
            differing = [k for k in a['initial_parameter_hashes'] if a['initial_parameter_hashes'][k] != b['initial_parameter_hashes'][k]]
            assert differing and not any('dfl.conv.weight' in k for k in differing)
            assert len(a['first_two_input_sha256_by_epoch']) == len(b['first_two_input_sha256_by_epoch']) == 100
            for left, right in zip(a['first_two_input_sha256_by_epoch'], b['first_two_input_sha256_by_epoch']):
                assert len(left) == len(right) == 2 and all(x != y for x, y in zip(left, right))
            comparisons.append({'model': model, 'seeds': [a['seed'], b['seed']],
                                'different_initial_parameter_names': differing,
                                'both_sampled_inputs_differ_at_all_100_epochs': True,
                                'raw_final_and_EMA_parameters_differ': True})
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--proof-directory', type=Path, required=True)
    args = parser.parse_args()
    assert (args.model is None) == (args.seed is None)
    torch.set_num_threads(1)
    args.proof_directory.mkdir(parents=True, exist_ok=True)
    bound = Bindings()
    p = verify_sources(args.root, bound)
    script_digest = bound.bind(__file__)
    targets = [(args.model, args.seed)] if args.model else list(itertools.product(MODELS, SEEDS))
    proofs = []
    for model, seed in targets:
        path = args.proof_directory / f'{model}_seed{seed}.json'
        if path.exists():
            proof = bound.read(path)
            assert proof['status'] == 'passed' and proof['script_sha256'] == script_digest
            assert proof['protocol_sha256'] == PROTOCOL_SHA256
            assert (proof['run']['model'], proof['run']['seed']) == (model, seed)
            for name, digest in proof['bound_input_sha256'].items():
                assert bound.bind(name) == digest
            bound.verify()
        else:
            own = Bindings()
            verify_sources(args.root, own)
            own.bind(__file__)
            run = audit_run(args.root, p, model, seed, own)
            proof = {'status': 'passed', 'utc': datetime.now(timezone.utc).isoformat(),
                     'script_sha256': script_digest, 'protocol_sha256': PROTOCOL_SHA256,
                     'run': run, 'bound_input_sha256': own.files,
                     'scope': 'Independent CPU reload, no inference or training; before/after byte checks.',
                     'limits': ['Two sampled inputs per epoch, not all inputs.',
                                'Finite saved states/losses do not imply finite gradients at every step.',
                                'New base actual counts do not reconstruct missing historical base counts.',
                                'Shared COCO pretraining; independent BDD fine-tuning only.']}
            with path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(proof, indent=2, allow_nan=False) + '\n')
        for name, digest in proof['bound_input_sha256'].items():
            assert bound.bind(name) == digest
        bound.bind(path)
        proofs.append(proof)
    bound.verify()
    result = {'status': 'individual_base_passed', 'runs': len(proofs)}
    if args.model is None:
        done = bound.read(args.root / 'BASE_TRAINING_COMPLETE.json')
        assert done['status'] == 'complete' and done['new_base_runs'] == 6
        result = {'status': 'all_six_completed_bases_passed', 'pairwise_checks': aggregate(proofs),
                  'protocol_sha256': PROTOCOL_SHA256, 'script_sha256': script_digest}
        bound.verify()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
