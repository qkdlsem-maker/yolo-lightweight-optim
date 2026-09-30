"""Reaggregate all 11 completed illumination records from per-image statistics."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class
from clean_followup_utils import sha, dump

torch.set_num_threads(2)
root = Path('review_20260929_clean')
assert json.loads((root / 'POSTPROCESS_COMPLETE.json').read_text())['status'] == 'complete'
models = json.loads((root / 'external_kitti/frozen_models.json').read_text())['models']
models = [r for r in models if r['group'].startswith('clean_')]
assert len(models) == 11
attrs_path = Path('review_20260929/legacy_val_attributes.json')
attrs = json.loads(attrs_path.read_text())
groups = ['all', 'daytime', 'night', 'dawn/dusk', 'undefined']
expected_images = [10000, 5258, 3929, 778, 35]
report = []
for model in models:
    folder = root / 'stratified' / model['name']
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert metrics['checkpoint_sha256'] == model['checkpoint_sha256']
    assert metrics['batch'] == 8 and metrics['imgsz'] == 640 and not metrics['fp16']
    with np.load(folder / 'image_statistics.npz', allow_pickle=False) as data:
        images = data['images'].tolist()
        assert len(images) == 10000 and set(images) == set(attrs)
        po, go = data['pred_offsets'], data['gt_offsets']
        assert len(po) == len(go) == 10001
        assert po[0] == go[0] == 0
        assert (np.diff(po) >= 0).all() and (np.diff(go) >= 0).all()
        assert len(data['tp']) == len(data['conf']) == len(data['pred_cls']) == po[-1]
        assert len(data['target_cls']) == go[-1] == 185526
        maximum_error = 0.
        for group, count in zip(groups, expected_images):
            selected = [i for i, name in enumerate(images) if group == 'all' or attrs[name]['timeofday'] == group]
            assert len(selected) == count
            pred_indices = np.concatenate([np.arange(po[i], po[i+1]) for i in selected])
            gt_indices = np.concatenate([np.arange(go[i], go[i+1]) for i in selected])
            inputs = {k: data[k][pred_indices] for k in ['tp', 'conf', 'pred_cls']}
            inputs['target_cls'] = data['target_cls'][gt_indices]
            calculated = ap_per_class(**inputs, plot=False)
            ap, ids = calculated[5], calculated[6].astype(int)
            saved = metrics['overall'] if group == 'all' else metrics['timeofday'][group]
            assert saved['images'] == count
            assert saved['instances_by_class'] == np.bincount(inputs['target_cls'].astype(int), minlength=10).tolist()
            values = dict(AP50_present_classes=float(ap[:, 0].mean()), AP50_95_present_classes=float(ap.mean()))
            if all(c in ids for c in range(9)):
                values['road9_AP50'] = float(np.mean([ap[list(ids).index(c), 0] for c in range(9)]))
                values['road9_AP50_95'] = float(np.mean([ap[list(ids).index(c)].mean() for c in range(9)]))
            else:
                assert saved['road9_AP50'] is None and saved['road9_AP50_95'] is None
            for key, value in values.items():
                error = abs(value - saved[key]); maximum_error = max(maximum_error, error)
                assert error < 1e-12, (model['name'], group, key)
            assert set(saved['class_metrics']) == {str(c) for c in ids}
            for index, category in enumerate(ids):
                for key, value in [('AP50', ap[index, 0]), ('AP50_95', ap[index].mean())]:
                    error = abs(value - saved['class_metrics'][str(category)][key])
                    maximum_error = max(maximum_error, error)
                    assert error < 1e-12
    report.append(dict(name=model['name'], metrics_sha256=sha(folder/'metrics.json'),
                       statistics_sha256=sha(folder/'image_statistics.npz'), statistics_bytes=(folder/'image_statistics.npz').stat().st_size,
                       all_five_groups_and_per_class_AP_reproduced=True, max_absolute_AP_error=maximum_error))
    print('REAGGREGATED', model['name'], maximum_error, flush=True)
dump(root / 'stratified_statistics_verification.json', dict(status='passed', models=11,
     attributes_sha256=sha(attrs_path), groups=groups, images_by_group=expected_images, runs=report))
