"""Synthetic CPU fixtures for source-to-summary integrity, not study outcomes."""
import copy,json,tempfile
from pathlib import Path
from verify_downloaded_results import verify_epoch_source,verify_metric_reference_parity

def rejected(function):
    try:function()
    except (ValueError,AssertionError):return
    raise AssertionError('Corrupt evidence accepted')

def run():
    source=dict(epoch=1,lr=.0001,det_loss=10.,feature_loss=.2,batches=1092,optimizer_steps_skipped=1,
        median_unclipped_grad_norm=23.,first_two_input_sha256=['a'*64,'b'*64],gradient_norm_finite_batches=1091,gradient_norm_nonfinite_batches=1,finite_gradient_clipped_batches=1091)
    summary=dict(source,gradient_median_unavailable_reason=None)
    verify_epoch_source(json.dumps(source),summary,False)
    native_null=copy.deepcopy(source);native_null['median_unclipped_grad_norm']=None
    native_null_summary=dict(native_null,gradient_median_unavailable_reason=None)
    verify_epoch_source(json.dumps(native_null),native_null_summary,False)
    bad=copy.deepcopy(native_null);bad['gradient_norm_nonfinite_batches']=0;bad['gradient_norm_finite_batches']=1092
    rejected(lambda:verify_epoch_source(json.dumps(bad),dict(bad,gradient_median_unavailable_reason=None),False))
    bad=copy.deepcopy(native_null);bad['gradient_norm_finite_batches']=1092
    rejected(lambda:verify_epoch_source(json.dumps(bad),dict(bad,gradient_median_unavailable_reason=None),False))
    changed=copy.deepcopy(summary);changed['optimizer_steps_skipped']=2
    rejected(lambda:verify_epoch_source(json.dumps(source),changed,False))
    old=copy.deepcopy(source);old['median_unclipped_grad_norm']=float('nan')
    old_summary=copy.deepcopy(summary);old_summary['median_unclipped_grad_norm']=None
    old_summary['gradient_median_unavailable_reason']='historical_NaN_median_does_not_identify_nonfinite_backward_count'
    verify_epoch_source(json.dumps(old),old_summary,True)
    rejected(lambda:verify_epoch_source(json.dumps(old),old_summary,False))
    old['det_loss']=float('inf')
    rejected(lambda:verify_epoch_source(json.dumps(old),old_summary,True))
    duplicate=json.dumps(source)[:-1]+',"epoch":1}'
    rejected(lambda:verify_epoch_source(duplicate,summary,False))
    class Fixture:
        def __init__(self):self.values={};self.endpoints=[];self.factors=[]
        def analysis(self,name):return self.endpoints if name=='ALL_84_ENDPOINTS' else self.factors
        def read(self,name,digest):return self.values[name]
    f=Fixture()
    for i in range(48):
        refs={};cells=[]
        for j,(policy,key) in enumerate((('source_ignore','kitti_source_ignore'),('no_ignore','kitti_no_ignore'))):
            value=.1+j*.1;classes={'person':value,'car':value};name=str(i)+'/'+key
            f.values[name]={'score':{'mean_AP50_95':value,'class_AP50_95':classes}}
            refs[key]={'artifact':name,'sha256':'a'*64}
            cells.append(dict(ignore_policy=policy,overlap_policy='iou_grid',recall_points=101,mean_AP=value,class_AP=dict(classes)))
        f.endpoints.append({'roster':{'endpoint_id':str(i)},'evidence':refs})
        f.factors.append({'endpoint_id':str(i),'cells':cells})
    assert verify_metric_reference_parity(f)==96
    f.factors[0]['cells'][0]['class_AP']['car']+=.001
    rejected(lambda:verify_metric_reference_parity(f))
    return dict(status='synthetic_source_integrity_checks_passed',scientific_results_checked=0,
        cases=['finite raw epoch equality','native explicit null with positive counter accepted','null without nonfinite batch rejected','inconsistent batch partition rejected','corrupt epoch count rejected','historical NaN median preserved with reason',
            'new NaN rejected','nonfinite outside historical median rejected','duplicate key rejected',
            '96 reference cells compared to source scores','changed class AP rejected'])

if __name__=='__main__':
    folder=Path(tempfile.mkdtemp(prefix='scie_source_integrity_'));result=run()
    (folder/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(folder);print(json.dumps(result))
