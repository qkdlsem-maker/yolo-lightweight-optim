"""Tiny fabricated CPU states: accept valid evidence, reject corrupted evidence.

No research checkpoint, image, annotation, cache, GPU, or real score is loaded.
"""
import copy
import csv
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from audit_completed_base import (Bindings, aggregate, audit_run, canonical_sha,
                                  finite, sha, tensor_sha, verify_sources)


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        head = torch.nn.Module()
        head.nc = 10
        head.dfl = torch.nn.Module()
        head.dfl.conv = torch.nn.Linear(2, 2, bias=False)
        self.model = torch.nn.ModuleList([torch.nn.Linear(2, 2), head])
        self.register_buffer('counter', torch.tensor(1))
        for p in self.parameters():
            p.data.fill_(1.)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fixture(root):
    name = 'nano_seed404'
    folder = root / 'base_records' / name
    folder.mkdir(parents=True)
    checkpoint = root / 'base' / name / 'weights' / 'last.pt'
    checkpoint.parent.mkdir(parents=True)
    source = root / 'prior' / 'pretrained' / 'yolov8n.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fabricated initial source binding')
    data = root / 'data.yaml'
    data.write_text('synthetic binding only; no loader is invoked')
    (root / 'base_train.py').write_text('synthetic source binding only')
    base = dict(epochs=100, batch={'nano': 64, 'large': 16}, nbs=64, imgsz=640, optimizer='SGD',
                lr0=.01, lrf=.01, cos_lr=False, momentum=.937, weight_decay=.0005, warmup_epochs=3.,
                patience=0, close_mosaic=10, amp=True, workers=8, deterministic=True,
                nano_source_sha256=sha(source))
    protocol = dict(base=base, previous_root=str(root / 'prior'),
                    data={'yaml': str(data), 'yaml_sha256': sha(data)})
    dump(root / 'protocol.json', protocol)
    model = FixtureModel()
    initial = {k: {'shape': list(v.shape), 'sha256': tensor_sha(v)} for k, v in model.named_parameters()}
    dump(folder / 'initial_parameters.json', initial)
    raw = copy.deepcopy(model.state_dict())
    ema = copy.deepcopy(raw)
    for k in raw:
        if 'dfl.conv.weight' not in k and raw[k].is_floating_point():
            raw[k].add_(.125)
            ema[k].add_(.25)
    model.load_state_dict(ema)
    torch.save({'model': model.half(), 'ema': None}, checkpoint)
    config = dict(name=name, model='nano', seed=404, epochs=100, batch=64, fraction=1.,
                  source=str(source), source_sha256=sha(source), data=str(data), data_sha256=sha(data),
                  script_sha256=sha(root / 'base_train.py'), protocol_sha256=sha(root / 'protocol.json'))
    dump(folder / 'config.json', config)
    effective = {k: v for k, v in base.items() if k not in ('batch', 'nano_source_sha256')}
    effective.update(seed=404, batch=64, fraction=1., data=str(data), cache=False, pretrained=True, save=True, val=True)
    dump(folder / 'effective_config.json', effective)
    for e in range(1, 101):
        dump(folder / f'epoch_{e:03d}.json', dict(epoch=e, loss_items=[1., 2., 3.],
             actual_step_counts={'attempted': 10*e, 'applied': 9*e, 'skipped': e}, native_EMA_updates=10*e,
             first_two_input_sha256=[hashlib.sha256(f'{e}/{b}'.encode()).hexdigest() for b in (0, 1)]))
    c = {'attempted': 1000, 'applied': 900, 'skipped': 100}
    state = dict(epoch=100, model=raw, ema=ema, actual_step_counts=c, ema_updates=1000,
                 optimizer={'state': {0: {'momentum_buffer': torch.ones(2)}}}, scaler={'scale': 128.},
                 scheduler={'last_epoch': 100}, python_rng=(3, (1, 2), None), numpy_rng=('fixture',),
                 torch_rng=torch.ones(4, dtype=torch.uint8), cuda_rng=torch.ones(4, dtype=torch.uint8))
    torch.save(state, folder / 'resume_state.pt')
    results = root / 'base' / name / 'results.csv'
    with results.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        writer.writerows({'epoch': e, 'loss': 1.} for e in range(1, 101))
    last = list(csv.DictReader(results.open()))[-1]
    done = dict(status='complete', epochs=100, model='nano', seed=404, finite_state=True,
                fixed_dfl_preserved=True, changed_parameter_tensors=2, checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), source_sha256=sha(source),
                initial_parameters_sha256=sha(folder / 'initial_parameters.json'),
                actual_step_counts=c, native_EMA_updates=1000, last_row=last)
    dump(folder / 'COMPLETE.json', done)
    return protocol, folder, checkpoint


def change_json(path, change):
    obj = read(path)
    change(obj)
    dump(path, obj)


def change_state(path, change):
    state = torch.load(path, map_location='cpu', weights_only=False)
    change(state)
    torch.save(state, path)


def main():
    torch.set_num_threads(1)
    passed = []
    cases = {
        'wrong_complete_seed': lambda r, f, c: change_json(f/'COMPLETE.json', lambda v: v.update(seed=505)),
        'wrong_config_lr': lambda r, f, c: change_json(f/'effective_config.json', lambda v: v.update(lr0=.001)),
        'wrong_data_bytes': lambda r, f, c: (r/'data.yaml').write_text('changed'),
        'checkpoint_bytes_changed': lambda r, f, c: c.write_bytes(c.read_bytes()+b'changed'),
        'wrong_epoch_identity': lambda r, f, c: change_json(f/'epoch_002.json', lambda v: v.update(epoch=1)),
        'missing_epoch': lambda r, f, c: (f/'epoch_050.json').rename(f/'preserved_missing_epoch.json'),
        'invalid_count_identity': lambda r, f, c: change_json(f/'epoch_010.json', lambda v: v['actual_step_counts'].update(skipped=11)),
        'nonmonotonic_counts': lambda r, f, c: change_json(f/'epoch_010.json', lambda v: v.update(actual_step_counts={'attempted': 10,'applied':9,'skipped':1},native_EMA_updates=10)),
        'EMA_incorrectly_counts_applied': lambda r, f, c: change_json(f/'epoch_010.json', lambda v: v.update(native_EMA_updates=90)),
        'missing_input_hash': lambda r, f, c: change_json(f/'epoch_010.json', lambda v: v.update(first_two_input_sha256=[])),
        'final_counts_disagree': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v.update(actual_step_counts={'attempted':1001,'applied':901,'skipped':100})),
        'raw_nonfinite': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v['model']['model.0.weight'].fill_(float('inf'))),
        'optimizer_nonfinite': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v['optimizer']['state'][0]['momentum_buffer'].fill_(float('nan'))),
        'checkpoint_EMA_disagrees': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v['ema']['model.0.weight'].add_(1.)),
        'fixed_DFL_changed': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v['model']['model.1.dfl.conv.weight'].add_(.125)),
        'raw_not_FP32': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v['model'].update({'model.0.weight':v['model']['model.0.weight'].half()})),
        'missing_RNG': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: v.pop('cuda_rng')),
        'raw_unchanged': lambda r, f, c: change_state(f/'resume_state.pt', lambda v: [x.fill_(1.) for x in v['model'].values() if x.is_floating_point()]),
    }
    with tempfile.TemporaryDirectory(prefix='completed_base_audit_fixture_') as tmp:
        root = Path(tmp) / 'valid'
        p, folder, checkpoint = fixture(root)
        good = audit_run(root, p, 'nano', 404, Bindings())
        assert good['actual_step_counts'] == {'attempted':1000,'applied':900,'skipped':100}
        assert good['native_EMA_updates'] == 1000 and good['raw_changed_parameter_tensors'] == 2
        passed.append('valid_raw_EMA_count_skip_and_saved_precision_binding')
        for label, corrupt in cases.items():
            r = Path(tmp) / label
            p, f, c = fixture(r)
            corrupt(r, f, c)
            try:
                audit_run(r, p, 'nano', 404, Bindings())
            except AssertionError:
                passed.append(label)
            else:
                raise AssertionError(f'corrupt evidence accepted: {label}')
        bound = Bindings()
        bound.bind(root/'data.yaml')
        (root/'data.yaml').write_text('changed after binding')
        try:
            bound.verify()
        except AssertionError:
            passed.append('mutation_after_first_read_rejected')
        else:
            raise AssertionError('input mutation accepted')
        try:
            verify_sources(root, Bindings())
        except AssertionError:
            passed.append('production_protocol_digest_gate_rejects_synthetic_protocol')
        else:
            raise AssertionError('fabricated protocol accepted in production gate')
        proofs = []
        for model in ('nano', 'large'):
            for seed in (404, 505, 606):
                row = copy.deepcopy(good)
                row.update(model=model, seed=seed)
                row['initial_parameter_hashes']['model.0.weight'] = hashlib.sha256(f'{model}/{seed}'.encode()).hexdigest()
                row['initial_parameter_digest'] = canonical_sha(row['initial_parameter_hashes'])
                row['raw_final_parameter_digest'] = f'raw-{model}-{seed}'
                row['EMA_parameter_digest'] = f'ema-{model}-{seed}'
                row['first_two_input_sha256_by_epoch'] = [[f'{seed}/{e}/0',f'{seed}/{e}/1'] for e in range(100)]
                proofs.append({'run':row})
        assert len(aggregate(proofs)) == 6
        passed.append('six_pairwise_seed_and_input_comparisons')
        for label, altered in [('missing_run', proofs[:-1]), ('duplicate_run', proofs[:-1]+[proofs[0]])]:
            try:
                aggregate(altered)
            except AssertionError:
                passed.append(label)
            else:
                raise AssertionError(label)
        altered = copy.deepcopy(proofs)
        altered[1]['run']['first_two_input_sha256_by_epoch'][55][1] = altered[0]['run']['first_two_input_sha256_by_epoch'][55][1]
        try:
            aggregate(altered)
        except AssertionError:
            passed.append('one_of_two_input_hashes_matching_rejected')
        else:
            raise AssertionError('partial input equality accepted')
    assert finite(torch.tensor([1.])) and not finite({'x':float('inf')})
    print(json.dumps({'status':'passed','utc':datetime.now(timezone.utc).isoformat(),
                      'cases':passed,'count':len(passed),'test_sha256':sha(__file__),
                      'audit_sha256':sha(Path(__file__).with_name('audit_completed_base.py')),
                      'scope':'Fabricated CPU states only; no completed research base audited.'},indent=2))


if __name__ == '__main__':
    main()
