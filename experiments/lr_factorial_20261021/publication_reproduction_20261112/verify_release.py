"""Read-only verification of numerical supplement against bound research evidence."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile
from verify_downloaded_results import Bundle, require, run

EXPECTED_DATASET = '1f7d0fc1e97af9536e0b6a90a82b8043f99d1a7cab8ad2e1b84044db7fb043fa'
EXPECTED_INDEX = 'c50308c238bd2fb774af6c9e3dd15cd5941ffb393dc00a45f10706ce61ac75d7'

def verify(dataset, evidence):
    require(hashlib.sha256(Path(dataset).read_bytes()).hexdigest() == EXPECTED_DATASET,
            'Unexpected numerical dataset bytes')
    evidence = Path(evidence)
    require(hashlib.sha256((evidence / 'INDEX.json').read_bytes()).hexdigest() == EXPECTED_INDEX,
            'Unexpected scientific evidence index')
    with zipfile.ZipFile(dataset) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
                    'Unsafe ZIP member')
        manifest = json.loads(archive.read('FILE_HASHES.json'))
        require(set(names) == {'FILE_HASHES.json'} | {item['path'] for item in manifest},
                'Supplement manifest does not cover exactly all members')
        for item in manifest:
            data = archive.read(item['path'])
            require(len(data) == item['bytes'], 'Supplement size differs: ' + item['path'])
            require(hashlib.sha256(data).hexdigest() == item['sha256'],
                    'Supplement hash differs: ' + item['path'])
        independent = run(evidence)
        bundle = Bundle(evidence)
        matched = []
        for name in names:
            if name.startswith('followup_results/') and name.endswith('.json'):
                analysis_name = name[len('followup_results/'):-len('.json')]
                require(json.loads(archive.read(name)) == bundle.analysis(analysis_name),
                        'Supplement and bound analysis disagree: ' + name)
                matched.append(name)
    return {
        'status': 'supplement_and_bound_arithmetic_checks_passed',
        'supplement_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'supplement_members_checked': len(manifest),
        'followup_analysis_files_identical_to_evidence': len(matched),
        'INDEX_sha256': EXPECTED_INDEX,
        'independent_verification': independent,
        'limits': 'No inference, native scoring or new research experiment is run; binary availability and reuse rights are not certified.'
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python verify_release.py S1_Dataset.zip /path/to/extracted_evidence')
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), indent=2, allow_nan=False))
