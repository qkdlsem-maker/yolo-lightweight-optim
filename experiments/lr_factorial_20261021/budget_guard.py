"""Enforce the fixed phase disk/GPU/wall budget without changing frozen training sources."""
import fcntl,os,signal,subprocess,time,traceback
from pathlib import Path
from job_runtime import R,read,dump,utc,budget_check,budget_used
lock=(R/'budget_guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def exact_live(pid,cmd):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')==[str(x).encode() for x in cmd]
 except FileNotFoundError:return False
def stop_owned():
 # Stop dispatchers first. Separate job sessions retain their own last atomic checkpoint.
 for name in ['launch.json','base_recovery_launch.json','continuation_launch.json','evaluation_launch.json','gradient_launch.json']:
  f=R/name
  if not f.exists():continue
  x=read(f)
  if exact_live(x['pid'],x['command']) and str(R) in ' '.join(x['command']):os.kill(x['pid'],signal.SIGTERM)
 killed=[]
 for f in (R/'attempts').glob('*.json'):
  x=read(f)
  if x['status']=='running' and exact_live(x['pid'],x['command']) and str(R) in ' '.join(x['command']):
   assert os.getpgid(x['pid'])==x['pid'],'Never signal an unowned process group'
   os.killpg(x['pid'],signal.SIGTERM);killed.append((f,x))
 for f,x in killed:
  deadline=time.time()+30
  while exact_live(x['pid'],x['command']) and time.time()<deadline:time.sleep(1)
  if exact_live(x['pid'],x['command']):os.killpg(x['pid'],signal.SIGKILL)
  # Parent may have stopped before closing its attempt record; preserve it, then close the charge.
  current=read(f)
  if 'ended_unix' not in current:
   dump(R/'budget_stop_attempts'/f.name,current)
   current.update(status='budget_stopped',ended_unix=time.time(),ended_utc=utc(),stop_reason='Fixed budget guard')
   dump(f,current)
while not (R/'PHASE_COMPLETE.json').exists():
 try:
  assert not (R/'BUDGET_STOP.json').exists(),'Budget stop already latched'
  budget_check();p=read(R/'protocol.json')['execution']
  used=int(subprocess.check_output(['du','-s','-B1',str(R)],text=True).split()[0])/2**30
  assert used<p['max_new_disk_GiB'],f'Phase disk budget reached: {used:.3f} GiB'
  hours,days=budget_used();dump(R/'budget_status.json',{'utc':utc(),'allocated_GPU_hours':hours,'wall_days':days,'phase_disk_GiB':used,'limits':p})
 except BaseException as error:
  dump(R/'BUDGET_STOP.json',{'utc':utc(),'reason':repr(error),'traceback':traceback.format_exc(),'last_atomic_training_states_preserved':True})
  stop_owned();raise
 time.sleep(30)
