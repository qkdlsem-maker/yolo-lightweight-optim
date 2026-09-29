"""Full train/validation comparison to the archived BDD100K 2018 JSON mirror.

Read-only for the existing experiment data. Uses normalized coordinates rounded
to six decimals; row order is ignored and duplicate multiplicities are retained.
"""
import json,zipfile,hashlib,io,time
from pathlib import Path
from collections import Counter
R=Path('work/reference_labels')
NAMES=['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train'];IDS={n:i for i,n in enumerate(NAMES)}
def objects(stream):
 decoder=json.JSONDecoder();buffer='';pos=0;eof=False;started=False
 while True:
  if len(buffer)-pos<2 and not eof:
   chunk=stream.read(1024*1024);eof=not chunk;buffer=buffer[pos:]+chunk;pos=0
  while pos<len(buffer) and buffer[pos].isspace():pos+=1
  if not started:
   assert buffer[pos]=='[';pos+=1;started=True;continue
  while pos<len(buffer) and (buffer[pos].isspace() or buffer[pos]==','):pos+=1
  if pos<len(buffer) and buffer[pos]==']':return
  try:item,end=decoder.raw_decode(buffer,pos)
  except json.JSONDecodeError:
   if eof:raise
   chunk=stream.read(1024*1024);eof=not chunk;buffer=buffer[pos:]+chunk;pos=0;continue
  yield item;pos=end
  if pos>1024*1024:buffer=buffer[pos:];pos=0
def canonical(category,values):return (category,)+tuple(round(float(v),6) for v in values)
report={}
for split in ['train','val']:
 start=time.time();counts=Counter();classes=Counter();extras_by_class=Counter();extra_shapes=Counter();examples=[];attrs={};source_names=set();source_digest=hashlib.sha256()
 if split=='train':
  original=zipfile.ZipFile(R/'legacy_train.zip');raw=original.open(original.namelist()[0])
 else:original=None;raw=(R/'legacy_val.json').open('rb')
 # Compute source checksum without holding the 1.45 GB training JSON in RAM.
 while b:=raw.read(1024*1024):source_digest.update(b)
 raw.close()
 raw=original.open(original.namelist()[0]) if original else (R/'legacy_val.json').open('rb')
 with io.TextIOWrapper(raw,encoding='utf-8') as stream,zipfile.ZipFile(R/f'local_{split}_labels.zip') as local,zipfile.ZipFile(R/f'legacy_{split}_reconstructed.zip','w',compression=zipfile.ZIP_DEFLATED) as clean:
  local_names={Path(n).stem for n in local.namelist()}
  for item in objects(stream):
   stem=Path(item['name']).stem;source_names.add(stem);attrs[stem]=item['attributes'];expected=[]
   for b in item.get('labels',[]):
    if b['category'] not in IDS or 'box2d' not in b:continue
    box=b['box2d'];x1,y1,x2,y2=[box[k] for k in ['x1','y1','x2','y2']]
    c=IDS[b['category']];v=canonical(c,[(x1+x2)/2560,(y1+y2)/1440,(x2-x1)/1280,(y2-y1)/720]);expected.append(v);classes[str(c)]+=1
   e=Counter(expected)
   text=local.read(stem+'.txt').decode() if stem in local_names else ''
   a=Counter(canonical(int(s.split()[0]),s.split()[1:]) for s in text.splitlines() if s.strip())
   matched=a&e;extra=a-e;missing=e-a
   positive=Counter({k:v for k,v in a.items() if k[3]>0 and k[4]>0})
   counts.update(images=1,images_exact=int(a==e),images_positive_exact=int(positive==e),local_boxes=sum(a.values()),source_boxes=sum(e.values()),matched_boxes=sum(matched.values()),local_only=sum(extra.values()),source_only=sum(missing.values()),local_nonpositive_boxes=sum(v for k,v in a.items() if k[3]<=0 or k[4]<=0))
   for row,n in extra.items():extras_by_class[str(row[0])]+=n;extra_shapes['nonpositive' if row[3]<=0 or row[4]<=0 else 'positive']+=n
   if (extra or missing) and len(examples)<12:examples.append({'image':stem,'local_only':list(extra.elements())[:4],'source_only':list(missing.elements())[:4]})
   clean.writestr(stem+'.txt','\n'.join(str(row[0])+' '+' '.join(f'{x:.6f}' for x in row[1:]) for row in expected))
   if counts['images']%10000==0:print(split,counts['images'],'images compared',flush=True)
 unmatched=[];unmatched_classes=Counter();unmatched_boxes=0
 with zipfile.ZipFile(R/f'local_{split}_labels.zip') as local:
  for stem in sorted(local_names-source_names):
   a=[canonical(int(s.split()[0]),s.split()[1:]) for s in local.read(stem+'.txt').decode().splitlines() if s.strip()]
   unmatched_boxes+=len(a);unmatched_classes.update(str(row[0]) for row in a)
   unmatched.append({'image':stem,'boxes':len(a),'nonpositive_boxes':sum(row[3]<=0 or row[4]<=0 for row in a)})
 if original:original.close()
 (R/f'legacy_{split}_attributes.json').write_text(json.dumps(attrs),encoding='utf-8')
 report[split]={'counts':dict(counts),'filename_sets_equal':local_names==source_names,'unmatched_local_images':unmatched,'unmatched_local_boxes':unmatched_boxes,'unmatched_local_class_counts':dict(unmatched_classes),'source_class_counts':dict(classes),'extra_class_counts':dict(extras_by_class),'extra_shapes':dict(extra_shapes),'source_json_sha256':source_digest.hexdigest(),'examples':examples,'seconds':time.time()-start}
 print(split,{k:v for k,v in report[split].items() if k not in ['examples','unmatched_local_images']},flush=True)
report['comparison_rule']='Class IDs and normalized cx,cy,w,h to six decimal places, ignoring row order, retaining duplicate multiplicities; image size 1280x720.'
report['source']='SoleSensei Kaggle mirror of BDD100K 2018 JSON annotations, version predating the 2020 revision. This is a full comparison to retrieved JSON, not a signed authenticity certificate from the original host.'
(R/'legacy_full_audit.json').write_text(json.dumps(report,indent=2))
