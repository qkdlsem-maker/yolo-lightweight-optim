"""Reproduce native AP from the released statistics, without loading any model."""
from pathlib import Path
import argparse,ast,hashlib,io,json,platform,zipfile
import numpy as np

EXPECTED_METRICS_SHA256='5202c6b9d5a9e7bb970cc94c509cb882ccfa71f56372aeed6261deb68dd3aaf3'
def require(condition,message):
    if not condition:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('Native_BDD_statistics_108.zip'))
    parser.add_argument('--metrics-source',type=Path,default=Path('evaluator_sources/metrics.py'))
    parser.add_argument('--report',type=Path,default=Path('native_AP_replay_report.json'))
    args=parser.parse_args()
    require(np.__version__=='1.26.4','Use NumPy 1.26.4 to reproduce the recorded calculation.')
    source=args.metrics_source.read_bytes()
    require(digest(source)==EXPECTED_METRICS_SHA256,'Recorded metrics source SHA-256 mismatch')
    module=ast.parse(source)
    selected=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in {'smooth','compute_ap','ap_per_class'}]
    require(len(selected)==3,'Unexpected recorded metrics source')
    namespace={'np':np,'Path':Path}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(args.metrics_source),'exec'),namespace)
    rows=[]
    with zipfile.ZipFile(args.archive) as z:
        manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))
        files={x['member']:x for x in manifest['files']}
        require(len(files)==len(manifest['files']),'Duplicate archive member')
        def verified(name):
            raw=z.read(name);record=files[name]
            require(len(raw)==record['bytes'] and digest(raw)==record['sha256'],'Member hash mismatch: '+name)
            return raw
        groups={}
        for name in files:
            if '/native_stats/' in name and name.endswith('.npz'):
                groups.setdefault(name.split('/',1)[0],[]).append(name)
        require(len(groups)==108,'Expected 108 distinct evaluated endpoints')
        for eid,names in sorted(groups.items()):
            require(len(names)==313,'Expected 313 statistic batches: '+eid)
            complete=json.loads(verified(eid+'/COMPLETE.json'))
            score_raw=verified(eid+'/score.json')
            require(digest(score_raw)==complete['files']['score.json']['sha256'],'Native score binding mismatch')
            score=json.loads(score_raw)
            values={key:[] for key in ['tp','conf','pred_cls','target_cls']}
            for name in sorted(names):
                raw=verified(name)
                require(digest(raw)==complete['files'][name.split('/',1)[1]]['sha256'],'Native statistic binding mismatch')
                with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                    for key in values:values[key].append(data[key])
            values={key:np.concatenate(parts,axis=0) for key,parts in values.items()}
            result=namespace['ap_per_class'](**values,plot=False)
            ap,classes=result[5],result[6]
            wanted=np.asarray(score['class_AP_by_IoU'],dtype=np.float64)
            require(classes.tolist()==score['AP_class_indices'],'Class index mismatch: '+eid)
            require(np.array_equal(ap,wanted),'Class-by-IoU AP mismatch: '+eid)
            require(len(values['target_cls'])==score['native_boxes'],'Target count mismatch: '+eid)
            rows.append({'endpoint_id':eid,'statistics_files':len(names),'class_iou_values':int(ap.size),
                         'all_AP_values_exact':True,'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean())})
            if len(rows)%10==0:print('Verified',len(rows),'endpoints',flush=True)
    report={'status':'passed','numpy':np.__version__,'python':platform.python_version(),
            'metrics_source_sha256':EXPECTED_METRICS_SHA256,'endpoint_count':len(rows),
            'statistics_files':sum(x['statistics_files'] for x in rows),
            'class_iou_values':sum(x['class_iou_values'] for x in rows),'all_values_exact':True,
            'model_loading':False,'inference_or_training':False,'endpoints':rows}
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('All 10,800 native AP values exactly reproduced.',flush=True)
if __name__=='__main__':main()
