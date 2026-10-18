"""Immutable, batch-atomic prediction archives. This module does no inference.

The caller supplies verified checkpoint/protocol/source/runtime bindings and a
content-hashed image manifest. A committed batch is never recomputed or replaced.
Interrupted staging directories are preserved, but never counted as predictions.
Use as a context manager: a kernel file lock excludes concurrent archive writers.
"""
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from prediction_io import INFERENCE, decode, encode, image_metadata


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected SHA256 hex digest')
    return value


def _durable_file(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def _sync_dir(path):
    # Linux supports directory fsync. On Windows os.rename is still atomic,
    # but this module does not claim equivalent power-loss durability.
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def make_manifest(bindings, images):
    required = {'endpoint_id', 'dataset_id', 'checkpoint_sha256', 'protocol_sha256',
                'source_sha256', 'runtime', 'inference'}
    if set(bindings) != required:
        raise ValueError('Missing or extra archive binding')
    if not all(isinstance(bindings[k], str) and bindings[k] for k in ('endpoint_id', 'dataset_id')):
        raise ValueError('Expected nonempty endpoint and dataset identifiers')
    for key in ('checkpoint_sha256', 'protocol_sha256'):
        _sha(bindings[key])
    if not bindings['source_sha256'] or not bindings['runtime']:
        raise ValueError('Source and runtime bindings are required')
    for value in bindings['source_sha256'].values():
        _sha(value)
    if bindings['inference'] != INFERENCE:
        raise ValueError('Inference settings differ from fixed protocol')
    checked = []
    for im in images:
        if set(im) != {'id', 'file_name', 'width', 'height', 'sha256'}:
            raise ValueError('Expected image metadata and input content SHA256')
        checked.append({**image_metadata(im), 'sha256': _sha(im['sha256'])})
    if not checked or len({x['id'] for x in checked}) != len(checked) or len({x['file_name'] for x in checked}) != len(checked):
        raise ValueError('Empty or duplicate image manifest')
    # Freeze the caller's dictionaries rather than retaining mutable aliases.
    return json.loads(canonical({'schema': 'prediction_archive_v1',
                                 'bindings': bindings, 'images': checked, 'batch_size': 32}))


class PredictionArchive:
    def __init__(self, root, bindings, images):
        self.root = Path(root)
        self.manifest = make_manifest(bindings, images)
        self.manifest_bytes = canonical(self.manifest)
        self.manifest_sha256 = digest(self.manifest_bytes)
        self.images = self.manifest['images']
        self.batch_count = (len(self.images) + 31) // 32
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0, os.SEEK_END)
                if self.lock.tell() == 0:
                    self.lock.write(b'0')
                    self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            self.lock = None
            raise RuntimeError('Prediction archive is already locked') from error
        try:
            self._initialize()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        if self.lock is not None:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def _guard(self):
        if self.lock is None:
            raise RuntimeError('Archive operations require the exclusive context lock')

    def _initialize(self):
        allowed = {'.lock', 'MANIFEST.json', 'batches', 'staging', 'COMPLETE.json'}
        if any(p.name not in allowed or p.is_symlink() for p in self.root.iterdir()):
            raise ValueError('Unexpected archive entry or symlink')
        manifest = self.root / 'MANIFEST.json'
        if manifest.exists():
            if manifest.read_bytes() != self.manifest_bytes:
                raise ValueError('Archive binding or input manifest mismatch; never mix runs')
        else:
            if any(p.name != '.lock' for p in self.root.iterdir()):
                raise ValueError('Uninitialized nonempty archive requires inspected recovery')
            _durable_file(manifest, self.manifest_bytes)
        for name in ('batches', 'staging'):
            (self.root / name).mkdir(exist_ok=True)
        _sync_dir(self.root)
        self.verify()

    def expected_images(self, index):
        if type(index) is not int or not 0 <= index < self.batch_count:
            raise ValueError('Invalid batch index')
        return self.images[index * 32:(index + 1) * 32]

    def _trace(self, trace, count):
        if len(trace) != 1:
            raise ValueError('One actual model forward per explicitly sliced batch required')
        item = trace[0]
        if set(item) != {'shape', 'dtype', 'device'} or item['shape'] != [count, 3, 640, 640] or item['dtype'] != 'torch.float32':
            raise ValueError('Actual model input differs from frozen geometry or precision')
        if not isinstance(item['device'], str) or not re.fullmatch(r'cpu|cuda(?::\d+)?', item['device']):
            raise ValueError('Unknown recorded inference device')
        return json.loads(canonical(trace))

    def _read_batch(self, index):
        expected = self.expected_images(index)
        folder = self.root / 'batches' / f'{index:06d}'
        marker_raw = (folder / 'COMMIT.json').read_bytes()
        marker = json.loads(marker_raw)
        required = {'schema', 'archive_manifest_sha256', 'batch_index', 'records', 'actual_forward_inputs'}
        if set(marker) != required or marker['schema'] != 'prediction_batch_v1' or marker['archive_manifest_sha256'] != self.manifest_sha256 or marker['batch_index'] != index:
            raise ValueError('Invalid batch commit binding')
        self._trace(marker['actual_forward_inputs'], len(expected))
        if len(marker['records']) != len(expected):
            raise ValueError('Missing or extra batch records')
        names = {f'{im["id"]:08d}.json' for im in expected} | {'COMMIT.json'}
        if {p.name for p in folder.iterdir()} != names or any(p.is_symlink() for p in folder.iterdir()):
            raise ValueError('Missing, extra, or symlinked prediction file')
        records = []
        for entry, im in zip(marker['records'], expected):
            name = f'{im["id"]:08d}.json'
            if set(entry) != {'file', 'sha256', 'image_sha256'} or entry['file'] != name or entry['image_sha256'] != im['sha256']:
                raise ValueError('Record/input binding mismatch')
            raw = (folder / name).read_bytes()
            if digest(raw) != _sha(entry['sha256']):
                raise ValueError('Prediction bytes changed')
            record = decode(raw)
            if record['image'] != image_metadata(im):
                raise ValueError('Prediction image metadata mismatch')
            records.append(record)
        return records, digest(marker_raw)

    def verify(self):
        self._guard()
        if (self.root / 'MANIFEST.json').read_bytes() != self.manifest_bytes:
            raise ValueError('Archive manifest changed')
        folders = sorted((self.root / 'batches').iterdir())
        if [p.name for p in folders] != [f'{i:06d}' for i in range(len(folders))] or len(folders) > self.batch_count or any(p.is_symlink() or not p.is_dir() for p in folders):
            raise ValueError('Noncontiguous or unexpected committed batch directory')
        hashes = [self._read_batch(i)[1] for i in range(len(folders))]
        complete = self.root / 'COMPLETE.json'
        if complete.exists():
            if len(hashes) != self.batch_count or complete.read_bytes() != canonical(self._completion(hashes)):
                raise ValueError('Invalid completion marker')
        return {'committed_batches': len(hashes), 'committed_images': min(len(hashes) * 32, len(self.images)),
                'batch_commit_sha256': hashes, 'complete': complete.exists(),
                'preserved_staging_entries': sorted(p.name for p in (self.root / 'staging').iterdir())}

    def commit(self, index, records, actual_forward_inputs):
        self._guard()
        # Full verification happens on entry/resume and completion; append checks
        # do not repeatedly rescan every previously committed prediction file.
        folders = sorted(p.name for p in (self.root / 'batches').iterdir())
        if folders != [f'{i:06d}' for i in range(index)] or (self.root / 'COMPLETE.json').exists():
            raise ValueError('Only the next uncommitted batch can be appended')
        expected = self.expected_images(index)
        if len(records) != len(expected):
            raise ValueError('Missing or extra predictions for this batch')
        trace = self._trace(actual_forward_inputs, len(expected))
        encoded = []
        for record, im in zip(records, expected):
            raw = encode(record)
            if decode(raw)['image'] != image_metadata(im):
                raise ValueError('Prediction metadata or order mismatch')
            encoded.append(raw)
        stage = self.root / 'staging' / f'{index:06d}-{uuid.uuid4().hex}'
        stage.mkdir()
        entries = []
        for raw, im in zip(encoded, expected):
            name = f'{im["id"]:08d}.json'
            _durable_file(stage / name, raw)
            entries.append({'file': name, 'sha256': digest(raw), 'image_sha256': im['sha256']})
        marker = {'schema': 'prediction_batch_v1', 'archive_manifest_sha256': self.manifest_sha256,
                  'batch_index': index, 'records': entries, 'actual_forward_inputs': trace}
        _durable_file(stage / 'COMMIT.json', canonical(marker))
        _sync_dir(stage)
        os.rename(stage, self.root / 'batches' / f'{index:06d}')
        _sync_dir(self.root / 'batches')
        _sync_dir(self.root / 'staging')

    def _completion(self, hashes):
        return {'schema': 'prediction_archive_complete_v1', 'manifest_sha256': self.manifest_sha256,
                'images': len(self.images), 'batches': self.batch_count, 'batch_commit_sha256': hashes}

    def finalize(self):
        state = self.verify()
        if state['committed_batches'] != self.batch_count:
            raise ValueError('Cannot finalize incomplete predictions')
        if not state['complete']:
            _durable_file(self.root / 'COMPLETE.json', canonical(self._completion(state['batch_commit_sha256'])))
            _sync_dir(self.root)
        return self.verify()

    def iter_records(self):
        if not self.verify()['complete']:
            raise ValueError('Scoring requires a complete verified prediction archive')
        for i in range(self.batch_count):
            yield from self._read_batch(i)[0]
