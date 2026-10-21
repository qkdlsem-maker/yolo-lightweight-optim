"""Start the fixed CPU audit queue only after the first full endpoint passes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('review_20261017_independent')
OUTPUT = ROOT/'final_evaluation_audits'
SOURCE = ROOT/'evaluation_tools/audit_final_evaluation.py'
SOURCE_SHA = 'abf66b83332814f890d58f18c4962e886e8e26625295820fbb136aa9fb3141e6'
PLAN_SHA = '9eb59cda78a1512a732b068cf172dd74009eb70232ddcbcceba32f6f82d5fc2d'
FIRST = 'base_nano_seed101_e100'


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new(name, record):
    with (OUTPUT/name).open('x') as f:
        json.dump(record, f, indent=2)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def main():
    first = read(OUTPUT/'first_launch.json')
    assert first['source_sha256'] == SOURCE_SHA == sha(SOURCE)
    assert sha(ROOT/'final_evaluation_plan.json') == PLAN_SHA
    while True:
        proc = Path('/proc',str(first['pid']),'cmdline')
        command = proc.read_bytes().decode().split('\0') if proc.exists() else []
        if command != first['command']+['']:
            break
        time.sleep(30)
    assert not list(OUTPUT.glob(FIRST+'_failure_*.json')), 'First endpoint failed; no further audits dispatched'
    proof = read(OUTPUT/(FIRST+'.json'))
    assert proof['status'] == 'passed' and proof['audit_source_sha256'] == SOURCE_SHA
    assert proof['plan_sha256'] == PLAN_SHA
    assert proof['endpoint_complete_sha256'] == sha(ROOT/'final_evaluation/endpoints'/FIRST/'ENDPOINT_COMPLETE.json')
    new('first_endpoint_validation.json', {'status':'passed','utc':utc(), 'proof_sha256':sha(OUTPUT/(FIRST+'.json')),
         'audit_source_sha256':SOURCE_SHA, 'scope':'First full real endpoint numerical replay passed; no inference repeated. Fixed remaining audit queue may proceed.'})
    command=[str(Path('venv/bin/python').absolute()),str(SOURCE),'--plan',str(ROOT/'final_evaluation_plan.json'),
             '--plan-sha256',PLAN_SHA,'--output',str(OUTPUT),'--watch']
    env=os.environ.copy()
    env.update(PYTHONPATH=str((ROOT/'evaluation_deps').absolute()), PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    assert sha(SOURCE) == SOURCE_SHA
    process=subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
    new('watch_child_launch.json', {'pid':process.pid, 'utc':utc(), 'command':command,'source_sha256':SOURCE_SHA})
    code=process.wait()
    if code:
        raise RuntimeError('Audit child stopped with code '+str(code)+'; inspect proof failure and log, do not automatically repeat')
    complete=read(OUTPUT/'AUDITS_COMPLETE.json')
    assert complete['status']=='passed' and len(complete['proof_sha256'])==48
    new('watch_finished.json', {'status':'passed','utc':utc(),'complete_sha256':sha(OUTPUT/'AUDITS_COMPLETE.json')})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        new('watch_failure_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json', {'utc':utc(),'traceback':traceback.format_exc(),'guard_source_sha256':sha(Path(__file__))})
        raise
