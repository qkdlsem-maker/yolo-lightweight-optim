"""Build an isolated validation view from reconstructed 2018 JSON labels."""
from pathlib import Path
import zipfile,json
project=Path.cwd();root=project/'review_20260929/verified_legacy'
images=root/'images/val';labels=root/'labels/val';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(project/'review_20260929/legacy_val_reconstructed.zip') as z:
 assert len(z.namelist())==10000
 for name in z.namelist():
  assert Path(name).name==name and name.endswith('.txt')
  target=labels/name;content=z.read(name)
  if target.exists():assert target.read_bytes()==content
  else:target.write_bytes(content)
  src=project/'data/raw/val/images'/Path(name).with_suffix('.jpg');assert src.is_file()
  dest=images/src.name
  if dest.exists():assert dest.resolve()==src.resolve()
  else:dest.symlink_to(src)
classes=['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train']
yaml='# Validation-only annotation sensitivity view. No training split.\npath: '+str(root)+'\ntrain: null\nval: images/val\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(classes))
(root/'data.yaml').write_text(yaml)
print('VERIFIED_LEGACY_VIEW_READY',len(list(labels.glob('*.txt'))),sum(len(p.read_text().splitlines()) for p in labels.glob('*.txt')),flush=True)
