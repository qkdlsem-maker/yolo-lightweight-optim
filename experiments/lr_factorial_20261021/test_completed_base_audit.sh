cd /home/futurenetworklab/choihyerim1129/yolo_lightweight_optim || exit 1
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 300 venv/bin/python - <<'PY'
import contextlib,io,json,runpy
from pathlib import Path
r=Path('review_20261021_lr_factorial')
output=r/'completed_base_audit_validation.json'
assert not output.exists(),'Do not overwrite a completed validation proof'
import sys
sys.path.insert(0,str(r.resolve()))
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):
 runpy.run_path(str(r/'test_completed_base_audit.py'),run_name='__main__')
proof=json.loads(buffer.getvalue())
assert proof['status']=='passed'
with output.open('x') as stream:stream.write(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
PY
