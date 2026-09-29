"""Synthetic tests of native KITTI matching and the R40 aggregation harness."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np

def line(cls,box,score=None,occ=0,trunc=0):
    text=f'{cls} {trunc} {occ} -10 '+' '.join(map(str,box))+' -1 -1 -1 -1000 -1000 -1000 -10'
    return text+(' '+str(score) if score is not None else '')+'\n'

def read_scores(folder):
    result={}
    for cls in ['car','pedestrian']:
        curve=np.loadtxt(folder/f'stats_{cls}_detection.txt')
        assert curve.shape==(3,41) and np.isfinite(curve).all() and ((curve>=0)&(curve<=1)).all()
        result[cls]=curve[:,1:].mean(axis=1).tolist()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='review_20260929_clean/external_kitti');a=p.parse_args();root=Path(a.root)
    exe=(root/'local_evaluator/evaluate_2d_local').resolve();testroot=root/'evaluator_synthetic_tests';testroot.mkdir(exist_ok=True)
    count=400;records={}
    car=(0,0,60,60);ped=(100,0,160,60);farcar=(300,0,360,60);farped=(400,0,460,60)
    cases=['perfect','missing','high_score_fp','dontcare','ignored_neighbor','half_missing','difficulty','low_height_detection']
    for case in cases:
        base=testroot/case;gt=base/'gt';det=base/'det';out=base/'out'
        for d in [gt,det,out]:d.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            g=line('Car',car)+line('Pedestrian',ped)
            d=line('Car',car,.9)+line('Pedestrian',ped,.9)
            if case=='missing' or case=='half_missing' and i%2:d=''
            if case in ['high_score_fp','dontcare','ignored_neighbor']:
                d+=line('Car',farcar,.99)+line('Pedestrian',farped,.99)
                if case=='dontcare':g+=line('DontCare',farcar)+line('DontCare',farped)
                if case=='ignored_neighbor':g+=line('Van',farcar)+line('Person_sitting',farped)
            if case=='difficulty':
                # Extra occluded GT is ignored in easy, counted as FN otherwise.
                g+=line('Car',farcar,occ=1)+line('Pedestrian',farped,occ=1)
            if case=='low_height_detection':
                d+=line('Car',(300,0,360,20),.99)+line('Pedestrian',(400,0,460,20),.99)
            (gt/f'{i:06d}.txt').write_text(g);(det/f'{i:06d}.txt').write_text(d)
        subprocess.run([str(exe),str(gt),str(det),str(out),str(count)],check=True)
        score=read_scores(out);records[case]=score
        if case in ['perfect','dontcare','ignored_neighbor','low_height_detection']:expected=[1.,1.,1.]
        elif case=='missing':expected=[0.,0.,0.]
        elif case in ['high_score_fp','half_missing']:expected=[.5,.5,.5]
        else:expected=[1.,.5,.5]
        for cls,values in score.items():assert np.allclose(values,expected,atol=1e-12,rtol=0),(case,cls,values,expected)
    result={'passed':True,'images_per_case':count,'cases':records,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'real_model_scores_observed':False}
    (root/'evaluator_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATOR_SYNTHETIC_TESTS_PASSED',flush=True)

if __name__=='__main__':main()
