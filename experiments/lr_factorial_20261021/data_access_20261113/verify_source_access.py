"""Anonymous read-only retrieval of the recorded BDD legacy annotation inputs."""
from pathlib import Path
import concurrent.futures, hashlib, json, time, urllib.request, zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'public_source_verification'
OUT.mkdir(exist_ok=True)
BASE = 'https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k/bdd100k_labels_release%2Fbdd100k%2Flabels%2Fbdd100k_labels_images_'
EXPECTED = {
    'val': 'c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af',
    'train': 'c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e',
}

def digest_stream(stream):
    h = hashlib.sha256(); count = 0
    while data := stream.read(4 * 1024**2):
        h.update(data); count += len(data)
    return h.hexdigest(), count

def verify(split):
    path = OUT / (split + '.download')
    url = BASE + split + '.json'
    row = {'split': split, 'source_url': url, 'authentication': 'none',
           'expected_json_sha256': EXPECTED[split], 'official_2020_identity_claimed': False}
    try:
        if not path.exists():
            start = time.monotonic()
            req = urllib.request.Request(url, headers={'User-Agent': 'Research-data-access-audit'})
            with urllib.request.urlopen(req, timeout=45) as source, path.with_suffix('.partial').open('xb') as dest:
                row['http_status'] = source.status
                row['content_type'] = source.headers.get('Content-Type')
                total = 0
                while data := source.read(4 * 1024**2):
                    total += len(data)
                    if total > 1700 * 1024**2 or time.monotonic() - start > 900:
                        raise ValueError('Transfer bound exceeded')
                    dest.write(data)
            path.with_suffix('.partial').rename(path)
        with path.open('rb') as f:
            row['download_sha256'], row['download_bytes'] = digest_stream(f)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                if len(members) != 1 or members[0].file_size > 1700 * 1024**2:
                    raise ValueError('Unexpected annotation archive layout')
                row['json_member'] = members[0].filename
                with archive.open(members[0]) as f:
                    row['actual_json_sha256'], row['json_bytes'] = digest_stream(f)
        else:
            row['actual_json_sha256'] = row['download_sha256']
            row['json_bytes'] = row['download_bytes']
        row['matches_research_source'] = row['actual_json_sha256'] == EXPECTED[split]
        row['status'] = 'verified' if row['matches_research_source'] else 'mismatch'
    except Exception as exc:
        row.update(status='failed', error_type=type(exc).__name__, message=str(exc)[:250])
    (OUT / (split + '_verification.json')).write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    return row

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(verify, ['val', 'train']))
    report = {'date': '2026-11-13', 'files': rows,
              'all_source_json_bytes_match': all(x.get('matches_research_source') for x in rows),
              'raw_annotations_publicly_redistributed_by_this_task': False}
    (OUT / 'ACCESS_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
