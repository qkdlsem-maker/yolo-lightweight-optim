"""Bind the already fixed evaluation images to file bytes, without model access.

Run from the experiment project root on Linux. Output filename/hash manifests
stay with the private experimental artifacts; publish only code and summary.
This checks content consistency, not official-host provenance or independence.
"""
import fcntl
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from PIL import Image

R = Path('review_20261017_independent')
VIEWS = R / 'common_annotations'
OUT = R / 'image_manifests'
PINNED = {
    'BUILD_COMPLETE.json': '58d224cc8038da89837e36e85564bade9ddb1007720e41cc0afd4a5afdab89db',
    'bdd_common.json': 'a3b1ad2918c8b27ec171a767d941a68580914edbcc36668b54e2eb4495a717fa',
    'kitti_source_ignore.json': 'bac59720f08481f3fa870afbca416f3c0d54741b6e023bd0b34f443df279a6c4',
    'kitti_no_ignore_sensitivity.json': 'c3d84f9504d04e24b3fb8bb9482e52d7458077809c0688f13b631cd60928f479',
}
DATASETS = {
    'bdd': {'annotation': 'bdd_common.json', 'count': 10000, 'suffix': '.jpg',
            'images': 'review_20260929_clean/clean_data/images/val'},
    'kitti': {'annotation': 'kitti_source_ignore.json', 'count': 7481, 'suffix': '.png',
              'images': 'review_20260929_clean/external_kitti/data/training/image_2'},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def save_new_or_identical(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Refusing to replace different existing manifest: ' + path.name)
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        # Preserve an interrupted partial file for inspected recovery.
        with temporary.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.rename(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'build.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in PINNED.items():
        if sha(VIEWS / name) != expected:
            raise ValueError('Fixed annotation view changed: ' + name)
    source_hash = sha(Path(__file__))
    completion = OUT / 'BUILD_COMPLETE.json'
    if completion.exists():
        record = json.loads(completion.read_bytes())
        assert record['builder_sha256'] == source_hash
        assert record['annotation_sha256'] == PINNED
        for name, item in record['outputs'].items():
            assert sha(OUT / name) == item['sha256']
        print('ALREADY_BUILT_OUTPUT_BYTES_VERIFIED_NO_MODEL_ACCESS')
        return
    datasets = {name: json.loads((VIEWS / config['annotation']).read_bytes())['images']
                for name, config in DATASETS.items()}
    other_kitti = json.loads((VIEWS / 'kitti_no_ignore_sensitivity.json').read_bytes())['images']
    assert datasets['kitti'] == other_kitti, 'Both KITTI metrics must consume identical image IDs/order'
    outputs, content_sets, dimensions, totals, duplicates, order_hashes = {}, {}, {}, {}, {}, {}
    for name, config in DATASETS.items():
        items = datasets[name]
        assert len(items) == config['count']
        assert [im['id'] for im in items] == list(range(1, config['count'] + 1))
        assert len({im['file_name'] for im in items}) == len(items)
        assert [im['file_name'] for im in items] == sorted(im['file_name'] for im in items)
        image_dir = Path(config['images'])
        assert {p.name for p in image_dir.glob('*' + config['suffix'])} == {im['file_name'] for im in items}
        rows, sizes, seen_hashes, total_bytes = [], Counter(), Counter(), 0
        for image in items:
            assert set(image) == {'id', 'file_name', 'width', 'height'}
            assert Path(image['file_name']).name == image['file_name']
            path = image_dir / image['file_name']
            before = path.stat()
            digest = sha(path)
            with Image.open(path) as im:
                assert im.size == (image['width'], image['height'])
                assert im.format == ('JPEG' if name == 'bdd' else 'PNG')
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            rows.append({**image, 'sha256': digest})
            seen_hashes[digest] += 1
            sizes[f'{image["width"]}x{image["height"]}'] += 1
            total_bytes += before.st_size
        artifact = name + '_images.json'
        outputs[artifact] = {**save_new_or_identical(OUT / artifact, encode(rows)), 'images': len(rows)}
        dimensions[name] = dict(sizes)
        content_sets[name] = set(seen_hashes)
        totals[name] = total_bytes
        duplicates[name] = {'duplicate_hash_groups': sum(n > 1 for n in seen_hashes.values()),
                            'extra_files_with_duplicate_bytes': sum(n - 1 for n in seen_hashes.values())}
        order_hashes[name] = hashlib.sha256(('\n'.join(x['file_name'] for x in items) + '\n').encode()).hexdigest()
        print(json.dumps({'dataset': name, 'images_hashed': len(rows), 'bytes_read': total_bytes}), flush=True)
    record = {
        'status': 'passed', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'builder_sha256': source_hash, 'annotation_sha256': PINNED, 'outputs': outputs,
        'image_order_sha256': order_hashes, 'image_bytes_read': totals,
        'image_dimensions': dimensions, 'within_partition_exact_byte_duplicates': duplicates,
        'BDD_validation_KITTI_exact_byte_shared_hashes': len(content_sets['bdd'] & content_sets['kitti']),
        'both_KITTI_annotation_variants_have_identical_images_and_order': True,
        'all_file_bytes_hashed_and_header_dimensions_checked': True,
        'original_data_or_training_modified': False, 'model_predictions_or_scores_accessed': False,
        'limits': 'Byte hashes pin the current image content; they do not authenticate the original download, rule out reencoded/near-duplicate scenes, check train/evaluation overlap or establish an untouched external test. BDD uses reused validation images; KITTI uses previously examined public labeled training images.',
    }
    save_new_or_identical(completion, encode(record))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
