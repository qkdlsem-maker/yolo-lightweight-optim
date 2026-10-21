"""Read-only CPU audit of one completed, prespecified continuation trajectory."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint['ema'] if checkpoint.get('ema') is not None else checkpoint['model']


def state_hash(state):
    # Same byte definition as the frozen trainer; not a file hash.
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('review_20261017_independent'))
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    r = args.root
    p = read(r / 'protocol.json')
    pre = read(r / 'PREFLIGHT_PASSED.json')
    assert sha(r / 'protocol.json') == pre['protocol_sha256']
    for name, digest in pre['source_sha256'].items():
        assert sha(r / name) == digest, name
    allowed = {}
    for seed in p['pair_seeds']:
        for arm, weight in [('control', 0.)] + [(a, w) for a in ['mse', 'cwd'] for w in p['continuation']['positive_weights']]:
            allowed[f'pair{seed}_{arm}_w{weight:g}'] = (seed, arm, weight)
    assert args.label in allowed
    seed, arm, weight = allowed[args.label]
    d = r / 'continuations' / args.label / f'{arm}_seed{seed+1000}_e40'
    assert (d / 'COMPLETE.json').exists(), 'Completed trajectories only'
    c = read(d / 'config.json')
    expected = {'arm': arm, 'seed': seed+1000, 'kd_weight': weight, 'epochs': 40,
                'batch': 64, 'workers': 4, 'lr': .0001, 'temperature': 4., 'smoke_batches': 0,
                'train_images': p['data']['train_images'], 'data': p['data']['yaml'],
                'script_sha256': pre['source_sha256']['continuation.py'],
                'protocol_sha256': pre['protocol_sha256']}
    for name, value in expected.items():
        assert c[name] == value, name
    bound = [r/'protocol.json', r/'continuation.py', Path(c['source']), Path(c['teacher']),
             Path(c['data']), d/'config.json', d/'COMPLETE.json', d/'parameter_delta.json', d/'training_state.pt']
    bound += sorted(d.glob('epoch_*.json'))
    for epoch in [10, 20, 40]:
        bound += [d/f'ema_epoch{epoch:02d}.pt', d/f'ema_epoch{epoch:02d}_metrics.json']
    before = {str(path): sha(path) for path in bound}
    assert before[c['source']] == c['student_sha256']
    assert before[c['teacher']] == c['teacher_sha256']
    assert before[c['data']] == p['data']['yaml_sha256']
    for arch, key in [('nano', 'source'), ('large', 'teacher')]:
        base = read(r/'base_records'/f'{arch}_seed{seed}'/'COMPLETE.json')
        assert base['checkpoint'] == c[key] and base['checkpoint_sha256'] == before[c[key]]

    source = model_from_checkpoint(c['source']).float()
    initial = source.state_dict()
    names = list(dict(source.named_parameters()))
    state = torch.load(d/'training_state.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == 40
    for key in ['student', 'adapters', 'ema', 'optimizer']:
        assert finite(state[key]), key
    assert all(state['student'][k].dtype == torch.float32 for k in names)
    differences = {k: float((state['student'][k]-initial[k]).abs().max()) for k in names}
    changed = sum(value > 0 for value in differences.values())
    assert changed > 0
    fixed = [k for k in names if 'dfl.conv.weight' in k]
    assert fixed and all(differences[k] == 0 for k in fixed)
    delta = read(d/'parameter_delta.json')
    assert delta['changed_tensors'] == changed and delta['fixed_dfl_preserved']
    assert math.isclose(delta['max_abs_delta'], max(differences.values()), rel_tol=1e-7)
    records = [read(d/f'epoch_{i:02d}.json') for i in range(1, 41)]
    assert len(list(d.glob('epoch_*.json'))) == 40
    assert [e['epoch'] for e in records] == list(range(1, 41))
    assert all(e['batches'] == math.ceil(p['data']['train_images']/64) for e in records)
    assert all(math.isfinite(e[k]) for e in records for k in ['det_loss', 'feature_loss', 'lr'])
    assert all(type(e['optimizer_steps_skipped']) is int and 0 <= e['optimizer_steps_skipped'] < e['batches'] for e in records)
    applied = sum(e['batches']-e['optimizer_steps_skipped'] for e in records)
    assert state['ema_updates'] == applied
    endpoints = []
    for epoch in [10, 20, 40]:
        path = d/f'ema_epoch{epoch:02d}.pt'
        metrics = read(d/f'ema_epoch{epoch:02d}_metrics.json')
        assert metrics['checkpoint_sha256'] == before[str(path)]
        assert all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ['map50', 'map50_95', 'precision', 'recall'])
        saved = model_from_checkpoint(path)
        assert saved.model[-1].nc == 10 and set(dict(saved.named_parameters())) == set(names)
        assert finite(saved.state_dict())
        assert any(not torch.equal(v.float(), initial[k]) for k, v in saved.named_parameters())
        final_binding = None
        if epoch == 40:
            assert metrics == read(d/'COMPLETE.json')
            assert state_hash(state['ema']) == metrics['state_sha256']
            for k, v in saved.state_dict().items():
                assert torch.equal(v, state['ema'][k].to(v.dtype)), k
            final_binding = True
        endpoints.append({'epoch': epoch, 'role': {10:'diagnostic_only', 20:'secondary', 40:'primary'}[epoch],
                          'checkpoint_sha256': before[str(path)],
                          'metrics_record_sha256': before[str(d/f'ema_epoch{epoch:02d}_metrics.json')],
                          'checkpoint_finite': True, 'ten_class_student_architecture': True,
                          'changed_from_starting_checkpoint': True,
                          'independently_matches_final_raw_ema_at_saved_precision': final_binding,
                          'saved_dfl_exact_starting_projection': all(torch.equal(saved.state_dict()[k].float(), initial[k]) for k in fixed)})
        del saved
    assert all(sha(path) == digest for path, digest in before.items()), 'Evidence changed during read-only audit'
    print(json.dumps({'status':'passed','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      'label':args.label,'pair_seed':seed,'data_seed':seed+1000,'arm':arm,'weight':weight,
      'epochs':40,'raw_float32_changed_parameter_tensors':changed,'raw_max_abs_parameter_delta':max(differences.values()),
      'raw_student_fixed_dfl_preserved':True,'finite_final_student_adapter_ema_optimizer_tensors':True,
      'optimizer_applied_updates':applied,'optimizer_skipped_updates':sum(e['optimizer_steps_skipped'] for e in records),
      'ema_update_count_matches_applied':True,'endpoints':endpoints,'bound_sha256':before,
      'scope':'Independent read-only CPU reload, no inference, training or score selection. Existing native diagnostic metrics are range-checked but do not substitute for the fixed final evaluations.',
      'limitations':['The final epoch40 EMA is independently bound to the raw sidecar and its recorded state digest. Epoch10/20 raw EMA sidecars were not retained; their saved checkpoints are bound to trainer metric records and checked for finite values/architecture.',
                     'Live frozen-teacher detachment was asserted by the trainer; this offline audit verifies teacher-file integrity, not historical GPU gradients.',
                     'Undefined gradient-norm medians remain in the original epoch records; per-batch gradient norms were not archived.']},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
