"""Validate all prespecified runs before computing descriptive paired effects."""
import argparse,csv,json,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='review_20260929');p.add_argument('--wait',action='store_true');args=p.parse_args()
root=Path(args.root);runs=root/'confirmatory';arms=['control','mse','cwd'];seeds=range(3)
expected=[f'{a}_seed{s}_e20' for a in arms for s in seeds]
while True:
 done=[n for n in expected if (runs/n/'COMPLETE.json').exists()]
 failures=[]
 for f in (root/'queue_logs').glob('worker*.jsonl'):
  for line in f.read_text().splitlines():
   x=json.loads(line)
   if x['exit_code'] or not x['complete']:failures.append(x)
 if failures:
  (root/'aggregation_status.json').write_text(json.dumps({'status':'technical_failure','failures':failures,'completed':done},indent=2));raise SystemExit('Queue failure; incomplete results not pooled')
 if len(done)==9:break
 if not args.wait:raise SystemExit(f'Pending: {len(done)}/9 completed; no treatment-effect aggregate')
 time.sleep(60)
configs={n:json.loads((runs/n/'config.json').read_text()) for n in expected}
metrics={n:json.loads((runs/n/'COMPLETE.json').read_text()) for n in expected}
for key in ['student_sha256','teacher_sha256','script_sha256','epochs','batch','lr','kd_weight','temperature']:
 assert len({c[key] for c in configs.values()})==1,key
for n in expected:
 delta=json.loads((runs/n/'parameter_delta.json').read_text())
 assert delta['changed_tensors']==183 and delta['fixed_dfl_preserved'],n
 assert metrics[n]['state_sha256']==json.loads((runs/n/'ema_epoch20_metrics.json').read_text())['state_sha256']
for epoch in range(1,21):
 byseed=[]
 for seed in seeds:
  hashes=[]
  for arm in arms:
   v=json.loads((runs/f'{arm}_seed{seed}_e20'/f'epoch_{epoch:02d}.json').read_text())
   assert v['batches']==1094
   hashes.append(tuple(v['first_two_input_sha256']))
  assert len(set(hashes))==1,('unpaired inputs',epoch,seed)
  byseed.append(hashes[0])
 assert len(set(byseed))==3,('repeated data seeds',epoch)
assert len({metrics[f'control_seed{s}_e20']['state_sha256'] for s in seeds})==3,'identical controls'
rows=[];summary={}
for arm in arms:
 for seed in seeds:
  m=metrics[f'{arm}_seed{seed}_e20'];c=metrics[f'control_seed{seed}_e20']
  rows.append({'arm':arm,'seed':seed,'AP50':m['map50'],'AP50_95':m['map50_95'],'delta_AP50_pp':100*(m['map50']-c['map50']),'delta_AP50_95_pp':100*(m['map50_95']-c['map50_95']),'state_sha256':m['state_sha256']})
 group=[v for v in rows if v['arm']==arm]
 summary[arm]={k:{'values':[v[k] for v in group],'mean':statistics.mean(v[k] for v in group),'sample_sd':statistics.stdev(v[k] for v in group)} for k in ['AP50','AP50_95','delta_AP50_pp','delta_AP50_95_pp']}
with (root/'confirmatory_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'status':'complete','primary_endpoint':'final epoch 20 EMA AP50:95','all_input_pairing_checks_passed':True,'independent_data_seeds':3,'independent_baseline_initializations':1,'summary':summary,'runs':rows,'limitations':['Three data seeds with one shared trained starting checkpoint','Repeatedly consulted validation set','Existing externally converted labels; test labels excluded','No equivalence or general superiority claim based solely on these descriptive differences']}
(root/'confirmatory_summary.json').write_text(json.dumps(result,indent=2))
(root/'aggregation_status.json').write_text(json.dumps({'status':'complete','completed':expected},indent=2))
print('ALL_NINE_RUNS_VERIFIED_AND_SUMMARIZED',flush=True)
