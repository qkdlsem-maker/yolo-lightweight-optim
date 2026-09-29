import json,zipfile
from pathlib import Path
R=Path('work/reference_labels');out={}
with zipfile.ZipFile(R/'local_val_labels.zip') as z:
 for n in z.namelist():
  classes=[]
  for s in z.read(n).decode().splitlines():
   row=s.split()
   if row and (float(row[3])<=0 or float(row[4])<=0):classes.append(int(row[0]))
  if classes:out[Path(n).stem]=classes
assert len(out)==51 and sum(map(len,out.values()))==52
(R/'zero_area_val_targets.json').write_text(json.dumps(out,indent=2))
print('Verified 52 zero-area validation targets in 51 images')
