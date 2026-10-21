"""All eight fixed cells from one complete archive; no inference or metric tuning."""
import argparse,fcntl,json,os,uuid
from pathlib import Path
from metric_factor_core import evaluate_grid,runtime_record
from prediction_archive import PredictionArchive,canonical,digest,sha256_file
from prediction_io import common_rows,image_metadata

SOURCES=('metric_factor_core.py','score_metric_factor_archive.py','prediction_archive.py','prediction_io.py','common_annotation_policy.json')
VARIANTS={'kitti_source_ignore':'source_ignore','kitti_no_ignore_sensitivity':'no_ignore'}

def write(path,value):
    with path.open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())

def check_sources(expected):
    if set(expected)!=set(SOURCES):raise ValueError('All fixed source/policy bindings are required')
    for name in SOURCES:
        if sha256_file(Path(__file__).parent/name)!=expected[name]:raise ValueError('Source changed: '+name)

def check_references(request):
    for group in ['annotation_views','reference_common_scores']:
        for item in request[group].values():
            if set(item)!= {'path','sha256'} or sha256_file(item['path'])!=item['sha256']:raise ValueError('Input reference changed: '+group)

def verify_output(output,manifest):
    if {p.name for p in output.iterdir()}!={'MANIFEST.json','COMPLETE.json','factorial.json'} or any(p.is_symlink() for p in output.iterdir()):
        raise ValueError('Missing, additional or symlinked output file')
    if (output/'MANIFEST.json').read_bytes()!=canonical(manifest):raise ValueError('Output request/runtime changed')
    expected={'schema':'metric_factor_archive_complete_v1','manifest_sha256':digest(canonical(manifest)),
              'factorial_sha256':sha256_file(output/'factorial.json'),'cells':8}
    if json.loads((output/'COMPLETE.json').read_bytes())!=expected:raise ValueError('Output bytes changed')
    return {'status':'complete','new_scores_computed':False,'cells':8,'manifest_sha256':expected['manifest_sha256']}

def check_common_cell(grid,reference):
    cell=grid['overlap_cells']['iou_grid']
    comparisons={'mean_AP50_95':cell['mean_AP'],'class_AP50_95':cell['class_AP'],
                 'precision_101':grid['precision_grid'],'max_recall_by_IoU':grid['max_recall_by_IoU']}
    for key in ['runtime','images','predictions','positive_ground_truth','crowd_ignore_ground_truth','class_order',
                'iou_thresholds','recall_thresholds','class_AP_by_IoU']:
        comparisons[key]=grid[key]
    for key,value in comparisons.items():
        if reference[key]!=value:raise ValueError('Common cell does not exactly reproduce saved reference: '+key)

def run(request,output):
    request=json.loads(canonical(request))
    required={'schema','dataset','archive_path','archive_manifest_sha256','archive_complete_sha256',
              'source_sha256','annotation_views','reference_common_scores'}
    if set(request)!=required or request['schema']!='metric_factor_archive_request_v1' or request['dataset']!='kitti':
        raise ValueError('Invalid fixed metric decomposition request')
    if any(set(request[k])!=set(VARIANTS) for k in ['annotation_views','reference_common_scores']):
        raise ValueError('Both annotation views and saved common references are mandatory')
    check_sources(request['source_sha256']);check_references(request)
    folder=Path(request['archive_path'])
    for name,key in [('MANIFEST.json','archive_manifest_sha256'),('COMPLETE.json','archive_complete_sha256')]:
        if sha256_file(folder/name)!=request[key]:raise ValueError('Archive binding changed')
    archived=json.loads((folder/'MANIFEST.json').read_bytes())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.with_name(output.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with PredictionArchive(folder,archived['bindings'],archived['images']) as archive:
            if not archive.verify()['complete']:raise ValueError('Prediction archive incomplete')
            images=[image_metadata(x) for x in archive.images];views={};references={}
            for variant in VARIANTS:
                data=json.loads(Path(request['annotation_views'][variant]['path']).read_bytes())
                if data['info']['annotation_variant']!=variant or data['info']['policy_sha256']!=request['source_sha256']['common_annotation_policy.json']:
                    raise ValueError('Wrong annotation policy or variant')
                if [image_metadata(x) for x in data['images']]!=images:raise ValueError('Image populations/order differ')
                if data['categories']!=[{'id':0,'name':'person'},{'id':2,'name':'car'}]:raise ValueError('Class mapping differs')
                views[variant]=data
                ref=json.loads(Path(request['reference_common_scores'][variant]['path']).read_bytes())
                if ref['schema']!='common_archive_score_v1' or ref['annotation_variant']!=variant:raise ValueError('Invalid common reference')
                if ref['annotation_sha256']!=request['annotation_views'][variant]['sha256']:raise ValueError('Common reference uses other labels')
                for key in ['archive_manifest_sha256','archive_complete_sha256']:
                    if ref[key]!=request[key]:raise ValueError('Common reference uses other predictions')
                references[variant]=ref
            ordinary=[{k:v for k,v in x.items() if k!='id'} for x in views['kitti_source_ignore']['annotations'] if x['iscrowd']==0]
            strict=[{k:v for k,v in x.items() if k!='id'} for x in views['kitti_no_ignore_sensitivity']['annotations']]
            if ordinary!=strict or any(x['iscrowd']!=0 for x in strict):raise ValueError('Ordinary targets differ')
            manifest={'schema':'metric_factor_archive_manifest_v1','request':request,'runtime':runtime_record(),'prediction_bindings':archived['bindings']}
            if output.exists():return verify_output(output,manifest)
            predictions=[];all_count=0
            for record in archive.iter_records():
                all_count+=len(record['detections']);predictions.extend(common_rows(record))
            projection=digest(canonical(predictions))
            for ref in references.values():
                if ref['shared_common_prediction_projection_sha256']!=projection or ref['mapped_person_car_predictions']!=len(predictions) or ref['all_class_predictions_preserved']!=all_count:
                    raise ValueError('Common reference projection/counts differ')
            stage=output.with_name(output.name+'.partial-'+uuid.uuid4().hex);stage.mkdir();write(stage/'MANIFEST.json',manifest)
            grids={};cells=[]
            # Reproduce BOTH prior common cells before computing 40-point variants.
            for n in [101,40]:
                for variant,policy in VARIANTS.items():
                    key=f'{policy}_r{n}';data=views[variant]
                    grid=evaluate_grid(data['images'],data['annotations'],predictions,n)
                    if n==101:check_common_cell(grid,references[variant]['score'])
                    grids[key]=grid
                    for overlap,values in grid['overlap_cells'].items():
                        cells.append({'ignore_policy':policy,'recall_points':n,'overlap_policy':overlap,'grid_reference':key,**values})
            artifact={'schema':'controlled_metric_factorial_archive_v1','cells':cells,'grids':grids,
                      'common_reference_cells_exactly_reproduced':list(VARIANTS),'native_KITTI_equivalence_claimed':False,
                      'archive_manifest_sha256':request['archive_manifest_sha256'],'archive_complete_sha256':request['archive_complete_sha256'],
                      'shared_common_prediction_projection_sha256':projection,'all_class_predictions_preserved':all_count,
                      'mapped_person_car_predictions':len(predictions),'difficulty_filter_applied':False}
            write(stage/'factorial.json',artifact)
            check_sources(request['source_sha256']);check_references(request);archive.verify()
            write(stage/'COMPLETE.json',{'schema':'metric_factor_archive_complete_v1','manifest_sha256':digest(canonical(manifest)),
                                        'factorial_sha256':sha256_file(stage/'factorial.json'),'cells':8})
            for path in [stage]:
                fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            os.rename(stage,output)
            fd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            result=verify_output(output,manifest);result['new_scores_computed']=True;return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(json.loads(a.request.read_bytes()),a.output),indent=2))
