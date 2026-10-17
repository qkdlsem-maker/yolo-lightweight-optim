"""Independent archive-to-output audit; intentionally does not import adapters."""
import hashlib,json,math,time,zipfile
from collections import defaultdict
from pathlib import Path

R=Path('review_20261017_independent');D=R/'common_annotations';OLD=Path('review_20260929_clean')
build=json.loads((D/'BUILD_COMPLETE.json').read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in build['source_sha256'].items():assert sha(R/n)==h,n
assert sha(OLD/'legacy_val_reconstructed.zip')==build['input_archive_sha256']['bdd']
assert sha(OLD/'external_kitti/data_object_label_2.zip')==build['input_archive_sha256']['kitti']
def load(name):
    assert sha(D/name)==build['outputs'][name]['sha256'],name
    data=json.loads((D/name).read_bytes());images=data['images'];rows=data['annotations']
    assert [r['id'] for r in rows]==list(range(1,len(rows)+1))
    assert [r['id'] for r in images]==list(range(1,len(images)+1))
    assert len({r['file_name'] for r in images})==len(images)
    grouped=defaultdict(list);valid={r['id'] for r in images}
    for row in rows:
        assert row['image_id'] in valid
        assert row['category_id'] in (0,2) and row['iscrowd'] in (0,1)
        x,y,w,h=row['bbox'];assert all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0
        assert math.isclose(row['area'],w*h,rel_tol=0,abs_tol=1e-8)
        grouped[row['image_id']].append(row)
    return data,grouped
def compare(got,expected):
    assert len(got)==len(expected)
    for actual,(category,crowd,bbox) in zip(got,expected):
        assert actual['category_id']==category and actual['iscrowd']==crowd
        assert all(math.isclose(a,b,rel_tol=0,abs_tol=1e-9) for a,b in zip(actual['bbox'],bbox))
bdd,grouped=load('bdd_common.json');source_bdd=0;target_bdd=0
with zipfile.ZipFile(OLD/'legacy_val_reconstructed.zip') as z:
    assert [Path(i['file_name']).stem+'.txt' for i in bdd['images']]==sorted(z.namelist())
    for image in bdd['images']:
        expected=[]
        for line in z.read(Path(image['file_name']).stem+'.txt').decode().splitlines():
            if not line.strip():continue
            c,*values=line.split();category=int(c);xc,yc,wn,hn=map(float,values);source_bdd+=1
            if category not in (0,2):continue
            w=wn*image['width'];h=hn*image['height']
            expected.append((category,0,[xc*image['width']-w/2,yc*image['height']-h/2,w,h]))
        compare(grouped[image['id']],expected);target_bdd+=len(expected)
a,ga=load('kitti_source_ignore.json');b,gb=load('kitti_no_ignore_sensitivity.json')
assert a['images']==b['images']
source_kitti=0;target_kitti=0;ignore_kitti=0
with zipfile.ZipFile(OLD/'external_kitti/data_object_label_2.zip') as z:
    for image in a['images']:
        expected_a=[];expected_b=[]
        assert image['file_name']==f"{image['id']-1:06d}.png"
        for line in z.read(f"training/label_2/{image['id']-1:06d}.txt").decode().splitlines():
            if not line.strip():continue
            fields=line.split();assert len(fields)==15;source_kitti+=1
            x1,y1,x2,y2=map(float,fields[4:8]);box=[x1,y1,x2-x1,y2-y1]
            label=fields[0]
            if label in ['Car','Pedestrian']:
                item=(2 if label=='Car' else 0,0,box);expected_a.append(item);expected_b.append(item)
            elif label=='Van':expected_a.append((2,1,box))
            elif label=='Person_sitting':expected_a.append((0,1,box))
            elif label=='DontCare':expected_a.extend([(0,1,box),(2,1,box)])
            else:assert label in ['Truck','Cyclist','Tram','Misc']
        compare(ga[image['id']],expected_a);compare(gb[image['id']],expected_b)
        target_kitti+=len(expected_b);ignore_kitti+=len(expected_a)-len(expected_b)
assert (source_bdd,target_bdd,source_kitti,target_kitti,ignore_kitti)==(185526,115768,51865,33229,25726)
record={'status':'passed','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'verification':'Independent implementation compares every serialized target/ignore row to pinned source archives, image IDs and counts; no adapter import',
 'verifier_sha256':sha(Path(__file__)),'build_manifest_sha256':sha(D/'BUILD_COMPLETE.json'),
 'bdd_source_rows_checked':source_bdd,'bdd_target_rows_checked':target_bdd,
 'kitti_source_rows_checked':source_kitti,'kitti_target_rows_checked':target_kitti,'kitti_compatibility_ignore_rows_checked':ignore_kitti,
 'real_model_predictions_or_scores_accessed':False,
 'limits':'Source archives remain legacy-mirror BDD and previously examined KITTI public training data; no new provenance or holdout-independence claim.'}
dest=D/'VERIFIED.json'
if dest.exists():raise FileExistsError('Verification already recorded; inspect it rather than overwrite')
dest.write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))
