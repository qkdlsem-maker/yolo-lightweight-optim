"""Independently verify frozen checkpoints and all serialized KITTI predictions."""
import hashlib
import json
import time
from pathlib import Path
from clean_followup_utils import sha, dump

root = Path('review_20260929_clean')
external = root / 'external_kitti'
frozen = json.loads((external / 'frozen_models.json').read_text())
rows = []
for model in frozen['models']:
    folder = external / 'results' / model['name']
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert sha(model['checkpoint']) == model['checkpoint_sha256'] == metrics['checkpoint_sha256']
    predictions = sorted((folder / 'data').glob('*.txt'))
    assert [p.stem for p in predictions] == [f'{i:06d}' for i in range(7481)]
    batches = sorted((folder / 'batches').glob('*.json'))
    assert len(batches) == 936
    expected_hashes = {}
    for index, batch in enumerate(batches):
        saved = json.loads(batch.read_text())
        assert list(saved) == [f'{i:06d}' for i in range(index*8, min(index*8+8, 7481))]
        expected_hashes.update(saved)
    combined = hashlib.sha256()
    total_bytes = detections = 0
    for path in predictions:
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected_hashes[path.stem]
        combined.update(path.name.encode() + b'\0' + data)
        total_bytes += len(data)
        for line in data.decode().splitlines():
            fields = line.split()
            assert len(fields) == 16 and fields[0] in ['Car', 'Pedestrian']
            detections += 1
    assert combined.hexdigest() == metrics['predictions_sha256']
    rows.append(dict(name=model['name'], checkpoint_bytes=Path(model['checkpoint']).stat().st_size,
                     images=7481, verified_batch_markers=len(batches), prediction_bytes=total_bytes,
                     detections=detections, predictions_sha256=combined.hexdigest()))
    print('VERIFIED', model['name'], flush=True)
dump(root / 'external_artifact_file_verification.json', dict(status='passed', models=len(rows),
     timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runs=rows))
