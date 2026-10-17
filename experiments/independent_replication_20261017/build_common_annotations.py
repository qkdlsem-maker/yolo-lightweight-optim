"""Build immutable normalized label views; never access model predictions.

Run from the authorized experiment project root after source-format unit tests.
Outputs contain source labels and stay private; publish only hashes and counts.
"""
import fcntl
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image
from common_annotation_adapters import bdd_rows, kitti_rows, identify_rows

R = Path('review_20261017_independent')
OLD = Path('review_20260929_clean')
OUT = R / 'common_annotations'
EXPECTED = {
    'bdd': 'df48ed99c5928e1a3d9c48e0b1c6e02f534e2d5c9bfdf3c7e323c617867db319',
    'kitti': '4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    raw = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == raw, f'Existing artifact differs: {path}'
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def summarize(rows):
    result = {'positive': Counter(), 'ignored': Counter(), 'duplicate_rows': 0}
    seen = Counter((r['image_id'], r['category_id'], r['iscrowd'], *r['bbox']) for r in rows)
    result['duplicate_rows'] = sum(n - 1 for n in seen.values())
    for row in rows:
        result['ignored' if row['iscrowd'] else 'positive'][str(row['category_id'])] += 1
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in result.items()}


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_names = ['common_annotation_policy.json', 'common_annotation_adapters.py', 'build_common_annotations.py']
    sources = {name: sha(R / name) for name in source_names}
    manifest_path = OUT / 'BUILD_COMPLETE.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_bytes())
        assert prior['source_sha256'] == sources
        for name, record in prior['outputs'].items():
            assert sha(OUT / name) == record['sha256']
        print('ALREADY_BUILT_AND_VERIFIED')
        return
    archives = {'bdd': OLD / 'legacy_val_reconstructed.zip',
                'kitti': OLD / 'external_kitti/data_object_label_2.zip'}
    for kind, path in archives.items():
        assert sha(path) == EXPECTED[kind], kind
    datasets, counts, dimensions, provenance = {}, {}, {}, {}
    categories = [{'id': 0, 'name': 'person'}, {'id': 2, 'name': 'car'}]
    with zipfile.ZipFile(archives['bdd']) as z:
        members = sorted(z.namelist())
        assert len(members) == 10000 and len(set(members)) == 10000
        assert all(Path(n).name == n and n.endswith('.txt') for n in members)
        images, labels, total, sizes = [], [], Counter(), Counter()
        for i, name in enumerate(members, 1):
            raw = z.read(name)
            assert (OLD / 'clean_data/labels/val' / name).read_bytes() == raw
            image_name = Path(name).stem + '.jpg'
            with Image.open(OLD / 'clean_data/images/val' / image_name) as im:
                width, height = im.size
            assert (width, height) == (1280, 720)
            sizes[f'{width}x{height}'] += 1
            converted, seen = bdd_rows(raw.decode('utf-8'), width, height)
            total.update(seen)
            images.append({'id': i, 'file_name': image_name, 'width': width, 'height': height})
            labels.extend(identify_rows(converted, i, len(labels) + 1))
        assert sum(total.values()) == 185526
        datasets['bdd_common.json'] = {'images': images, 'annotations': labels, 'categories': categories}
        counts['bdd_source_categories'] = dict(total)
        dimensions['bdd'] = dict(sizes)
        provenance['bdd_image_order_sha256'] = hashlib.sha256(('\n'.join(x['file_name'] for x in images) + '\n').encode()).hexdigest()
    with zipfile.ZipFile(archives['kitti']) as z:
        files = sorted(n for n in z.namelist() if not n.endswith('/'))
        expected = [f'training/label_2/{i:06d}.txt' for i in range(7481)]
        assert files == expected
        images, ignored_rows, strict_rows, total, sizes = [], [], [], Counter(), Counter()
        for i, name in enumerate(files, 1):
            raw = z.read(name)
            assert (OLD / 'external_kitti/data' / name).read_bytes() == raw
            image_name = Path(name).stem + '.png'
            with Image.open(OLD / 'external_kitti/data/training/image_2' / image_name) as im:
                width, height = im.size
            sizes[f'{width}x{height}'] += 1
            a, seen_a = kitti_rows(raw.decode('utf-8'), 'source_ignore_regions')
            b, seen_b = kitti_rows(raw.decode('utf-8'), 'no_ignore_sensitivity')
            assert seen_a == seen_b
            assert [r for r in a if not r['iscrowd']] == b
            total.update(seen_a)
            images.append({'id': i, 'file_name': image_name, 'width': width, 'height': height})
            ignored_rows.extend(identify_rows(a, i, len(ignored_rows) + 1))
            strict_rows.extend(identify_rows(b, i, len(strict_rows) + 1))
        assert dict(total) == {'Pedestrian': 4487, 'Truck': 1094, 'Car': 28742, 'Cyclist': 1627,
                               'DontCare': 11295, 'Misc': 973, 'Van': 2914, 'Tram': 511, 'Person_sitting': 222}
        assert len(strict_rows) == 33229 and len(ignored_rows) == 58955
        datasets['kitti_source_ignore.json'] = {'images': images, 'annotations': ignored_rows, 'categories': categories}
        datasets['kitti_no_ignore_sensitivity.json'] = {'images': images, 'annotations': strict_rows, 'categories': categories}
        counts['kitti_source_categories'] = dict(total)
        dimensions['kitti'] = dict(sizes)
        provenance['kitti_image_order_sha256'] = hashlib.sha256(('\n'.join(x['file_name'] for x in images) + '\n').encode()).hexdigest()
    outputs = {}
    for name, data in datasets.items():
        data['info'] = {'policy_sha256': sources['common_annotation_policy.json'],
                        'annotation_variant': name.removesuffix('.json')}
        output = save(OUT / name, data)
        output.update(images=len(data['images']), annotations=len(data['annotations']),
                      annotation_counts=summarize(data['annotations']))
        outputs[name] = output
    record = {'status': 'passed', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'source_sha256': sources, 'input_archive_sha256': EXPECTED, 'outputs': outputs,
              'source_category_counts': counts, 'image_dimensions': dimensions,
              'image_order_hashes': provenance, 'all_extracted_labels_match_archive_bytes': True,
              'all_image_headers_checked': True, 'positive_targets_identical_between_KITTI_variants': True,
              'original_data_unchanged': True, 'real_model_predictions_or_scores_accessed': False,
              'remaining': 'Independent conversion verification, inference serialization tests, frozen 48-endpoint manifest and actual model evaluation after training'}
    save(manifest_path, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
