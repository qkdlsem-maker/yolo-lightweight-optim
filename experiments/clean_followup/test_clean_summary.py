"""Exercise pooling with a known paired effect and reject missing/unpaired runs."""
from pathlib import Path
import json,tempfile,subprocess,sys
script=Path(__file__).resolve().parent/'summarize_clean_confirmatory.py'
with tempfile.TemporaryDirectory(dir=script.parent) as tmp:
 root=Path(tmp);cfg={k:1 for k in ['student_sha256','teacher_sha256','script_sha256','epochs','batch','lr','kd_weight','temperature']}
 cfg.update(epochs=20,batch=64,train_images=69863,data='clean_bdd.yaml')
 for a in ['control','mse','cwd']:
  for s in range(3):
   d=root/'confirmatory'/f'{a}_seed{s}_e20';d.mkdir(parents=True)
   v=.25+.01*s+{'control':0,'mse':.01,'cwd':-.01}[a]
   m={'map50':v+.2,'map50_95':v,'state_sha256':f'{a}{s}'}
   for n,x in [('config.json',cfg),('COMPLETE.json',m),('ema_epoch20_metrics.json',m),('parameter_delta.json',{'changed_tensors':183,'fixed_dfl_preserved':True})]:
    (d/n).write_text(json.dumps(x))
   for e in range(1,21):(d/f'epoch_{e:02d}.json').write_text(json.dumps({'batches':1092,'first_two_input_sha256':[f'{s}:{e}:0',f'{s}:{e}:1']}))
 command=[sys.executable,str(script),'--root',str(root)]
 good=subprocess.run(command,capture_output=True,text=True);assert good.returncode==0,good.stderr
 result=json.loads((root/'confirmatory_summary.json').read_text())
 assert abs(result['summary']['mse']['delta_AP50_95_pp']['mean']-1)<1e-10
 assert abs(result['summary']['cwd']['delta_AP50_95_pp']['mean']+1)<1e-10
 (root/'confirmatory/cwd_seed2_e20/COMPLETE.json').rename(root/'withheld.json')
 missing=subprocess.run(command,capture_output=True,text=True);assert missing.returncode!=0 and 'Pending' in missing.stderr
 (root/'withheld.json').rename(root/'confirmatory/cwd_seed2_e20/COMPLETE.json')
 bad_count=root/'confirmatory/mse_seed0_e20/epoch_01.json'
 original_count=bad_count.read_text();wrong=json.loads(original_count);wrong['batches']=1094;bad_count.write_text(json.dumps(wrong))
 count_failure=subprocess.run(command,capture_output=True,text=True);assert count_failure.returncode!=0 and 'unexpected minibatch count' in count_failure.stderr
 bad_count.write_text(original_count)
 bad=root/'confirmatory/cwd_seed0_e20/epoch_01.json';bad.write_text(json.dumps({'batches':1092,'first_two_input_sha256':['WRONG','INPUT']}))
 mismatch=subprocess.run(command,capture_output=True,text=True);assert mismatch.returncode!=0 and 'unpaired inputs' in mismatch.stderr
print('PASS: known paired effects, incomplete-run rejection, and input-mismatch rejection')
