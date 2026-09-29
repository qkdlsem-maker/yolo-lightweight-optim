"""Read-only exact image/label duplicate audit; no dataset changes."""
from pathlib import Path
import hashlib,json,itertools,time
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
r=Path('data/raw');out=Path('review_20260929')
def inspect(p):
 return p.parent.parent.name,p.name,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size
t=time.time();results={};details={}
for kind,pattern in [('images','*.jpg'),('labels','*.txt')]:
 rows=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for split in ['train','val','test']:
   part=list(pool.map(inspect,sorted((r/split/kind).glob(pattern))))
   rows.extend(part);print(kind,split,len(part),flush=True)
 bysplit={s:defaultdict(list) for s in ['train','val','test']}
 for s,n,h,size in rows:bysplit[s][h].append(n)
 overlap={};examples={}
 for a,b in itertools.combinations(bysplit,2):
  shared=sorted(set(bysplit[a]) & set(bysplit[b]));key=a+'_'+b
  overlap[key]={'distinct_shared_sha256':len(shared),'files_in_first':sum(len(bysplit[a][h]) for h in shared),'files_in_second':sum(len(bysplit[b][h]) for h in shared)}
  examples[key]=[{a:bysplit[a][h],b:bysplit[b][h],'sha256':h} for h in shared[:20]]
 results[kind]={'counts':{s:sum(len(n) for n in v.values()) for s,v in bysplit.items()},'distinct_hashes':{s:len(v) for s,v in bysplit.items()},'cross_split':overlap,'examples':examples,'total_bytes':sum(row[3] for row in rows)}
 with (out/(kind+'_sha256_manifest.jsonl')).open('w') as f:
  for s,n,h,size in rows:f.write(json.dumps({'split':s,'name':n,'sha256':h,'bytes':size})+'\n')
results['seconds']=time.time()-t
results['scope']='Exact encoded-file equality only. This does not establish annotation origin, visual near-duplicates, video-level independence, or official split identity.'
(out/'data_hash_audit.json').write_text(json.dumps(results,indent=2));print('HASH_AUDIT_COMPLETE',flush=True)
