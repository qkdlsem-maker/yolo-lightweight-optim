"""Independent arithmetic and roster review of a downloaded, hash-bound result set.

No training, inference, original scorer imports, or changes to experiment files.
Scalar checks do not reconstruct unavailable tensors or historical AMP gradients.
"""
import hashlib,itertools,json,math,statistics,sys
from fractions import Fraction
from pathlib import Path

PHASE='review_20261021_lr_factorial'
SEEDS=(101,202,303,404,505,606)
RATES=(.0001,.001)
ARMS=('control','mse','cwd')
PROTOCOL='8dac98e36fee058726b9475487896c49f53f246e960cdfc0ba7e6acb895cdc9d'
INTERPRETATION='ffea92e15e8f0d7973e6ccc96083227124c3eb42de25d746f4611b76471f02b8'

def require(ok,message):
    if not ok:raise ValueError(message)

def load(raw):
    def pairs(items):
        obj={}
        for key,value in items:
            require(key not in obj,'Duplicate JSON key: '+key);obj[key]=value
        return obj
    def bad(value):raise ValueError('Nonfinite JSON literal: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)

def close(actual,expected,label,tol=2e-11):
    require(type(actual) in (float,int) and math.isfinite(actual),label+' is not finite')
    require(math.isclose(actual,expected,rel_tol=tol,abs_tol=tol),label+' disagrees')

def frac(value):
    require(type(value) in (int,float) and math.isfinite(value),'Nonfinite numerical input')
    return Fraction(str(value))

def t975_df5():
    # Integrate the df5 density after x=sqrt(5)*tan(theta), independently of
    # the frozen analysis module's constant or any statistical package.
    def cdf(x):
        theta=math.atan(x/math.sqrt(5))
        return .5+theta/math.pi+2*math.sin(2*theta)/(3*math.pi)+math.sin(4*theta)/(12*math.pi)
    lo,hi=0.,10.
    for _ in range(80):
        mid=(lo+hi)/2
        if cdf(mid)<.975:lo=mid
        else:hi=mid
    return (lo+hi)/2

class Bundle:
    def __init__(self,root):
        self.root=Path(root).resolve();self.index=load((self.root/'INDEX.json').read_bytes())
        require(self.index['schema']=='fixed_result_export_v1','Export schema')
        rows=self.index['files'];self.rows={row['remote_path']:row for row in rows}
        require(len(self.rows)==len(rows),'Duplicate export path')
        self.checked={}
        for name,row in self.rows.items():
            path=(self.root/row['archive_path']).resolve()
            require(path.is_relative_to(self.root),'Export path traversal')
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            require(len(raw)==row['bytes'] and digest==row['sha256'],'Export bytes: '+name)
            self.checked[name]=digest
        require(self.checked[PHASE+'/protocol.json']==PROTOCOL,'Frozen protocol changed')
        require(self.checked[PHASE+'/protocol_interpretation.json']==INTERPRETATION,'Frozen interpretation changed')

    def read(self,name,expected=None):
        require(name in self.rows,'Absent exported artifact: '+name)
        if expected is not None:require(self.checked[name]==expected,'Evidence reference: '+name)
        return load((self.root/self.rows[name]['archive_path']).read_bytes())

    def analysis(self,name):return self.read(PHASE+'/analysis_final/'+name+'.json')

    def raw(self,name,expected):
        require(self.checked[name]==expected,'Raw evidence reference: '+name)
        raw=(self.root/self.rows[name]['archive_path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==expected,'Raw evidence changed: '+name)
        return raw

def extract_scores(values):
    """Separate implementation of the18 reporting definitions, from source JSON."""
    result={}
    for metric,key in [('AP50_95','metrics/mAP50-95(B)'),('AP50','metrics/mAP50(B)')]:
        result['bdd_ten_class_'+metric]=values['native_bdd']['results'][key]
    for source,prefix in [('bdd_common','bdd_common'),('kitti_source_ignore','kitti_common_source_ignore'),('kitti_no_ignore','kitti_common_no_ignore')]:
        source_score=values[source]['score'];result[prefix+'_AP50_95']=source_score['mean_AP50_95']
        for cls in ('person','car'):result[prefix+'_'+cls+'_AP50_95']=source_score['class_AP50_95'][cls]
    native=values['native_kitti'];result['kitti_native_mean_moderate_R40']=native['mean_moderate_AP_R40']
    for cls,difficulty in itertools.product(('car','pedestrian'),('easy','moderate','hard')):
        result['kitti_native_'+cls+'_'+difficulty+'_R40']=native['classes'][cls][difficulty]['AP_R40']
    require(len(result)==18 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in result.values()),'18 finite AP fractions')
    return result

def verify_summary(summary,values,interval,label):
    require(summary['expected_pairs']==summary['available_pairs']==len(values) and summary['complete'],label+' complete sample')
    n=len(values);mean=math.fsum(values)/n
    sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1))
    for key,expected in [('mean_pp',mean),('sample_sd_pp',sd),('minimum_pp',min(values)),('maximum_pp',max(values))]:close(summary[key],expected,label+' '+key)
    ci=summary['t_interval_95_pp']
    if interval:
        require(n==6 and ci['df']==5,label+' interval sample')
        half=t975_df5()*sd/math.sqrt(n)
        close(ci['lower'],mean-half,label+' CI lower');close(ci['upper'],mean+half,label+' CI upper')
    else:require(ci is None,label+' unexpected interval')

def contrast_terms(kind,arm,rate):
    if kind=='interaction':return [(.001,arm,1),(.001,'control',-1),(.0001,arm,-1),(.0001,'control',1)]
    if kind=='kd_minus_control':return [(rate,arm,1),(rate,'control',-1)]
    require(kind=='high_minus_low','Unknown contrast')
    return [(.001,arm,1),(.0001,arm,-1)]

def verify_endpoints_and_contrasts(bundle):
    rows=bundle.analysis('ALL_84_ENDPOINTS');plan=bundle.read(PHASE+'/factorial_endpoint_location_plan.json')
    by_id={row['roster']['endpoint_id']:row for row in rows}
    require(len(rows)==len(by_id)==84,'84 distinct follow-up endpoints')
    require({r['endpoint_id'] for r in plan['endpoints']}==set(by_id),'Endpoint plan identities')
    for row in plan['endpoints']:require(by_id[row['endpoint_id']]['roster']==row,'Original endpoint roster preserved')
    indexed={};reused=0;bases=[]
    for endpoint in rows:
        roster=endpoint['roster'];eid=roster['endpoint_id']
        source={key:bundle.read(ref['artifact'],ref['sha256']) for key,ref in endpoint['evidence'].items()}
        require(extract_scores(source)==endpoint['scores'],'Source score extraction: '+eid)
        audit=bundle.read(endpoint['audit']['path'],endpoint['audit']['sha256']);require(audit['status']=='passed','Endpoint audit: '+eid)
        reused+=roster['action']=='reuse_completed_evaluation_only'
        if roster['stage']=='base':bases.append(endpoint);continue
        key=(roster['pair_seed'],roster['lr'],roster['arm'],roster['epoch'])
        require(key not in indexed,'Duplicate continuation slot');indexed[key]=endpoint
    require(len(bases)==12 and reused==24,'Base/reuse counts')
    require(set(indexed)==set(itertools.product(SEEDS,RATES,ARMS,(20,40))),'Full72 continuation slots')
    require(bundle.analysis('BASE_REFERENCES_12')==bases,'Base references unchanged')
    contrast_count=pair_count=0;primary=[]
    expected_keys={(epoch,'interaction',arm,None) for epoch,arm in itertools.product((20,40),ARMS[1:])}
    expected_keys|={(epoch,'kd_minus_control',arm,rate) for epoch,arm,rate in itertools.product((20,40),ARMS[1:],RATES)}
    expected_keys|={(epoch,'high_minus_low',arm,None) for epoch,arm in itertools.product((20,40),ARMS)}
    for metric in rows[0]['scores']:
        report=bundle.analysis('contrasts/'+metric)
        require(report['protocol_sha256']==PROTOCOL and report['interpretation_sha256']==INTERPRETATION,'Contrast protocol')
        contrasts=report['contrasts'];require(len(contrasts)==18,'18 fixed contrasts per metric')
        require({(c['epoch'],c['kind'],c['arm'],c['learning_rate']) for c in contrasts}==expected_keys,'Contrast roster')
        source_rows=report['endpoint_rows'];require(len(source_rows)==72,'Contrast source row count')
        for r in source_rows:
            key=(r['pair_seed'],r['learning_rate'],r['arm'],r['epoch']);original=indexed[key]
            require(r['score']==original['scores'][metric] and r['endpoint_id']==original['roster']['endpoint_id'] and r['status']=='completed' and r['reason'] is None,'Contrast source row identity')
        for contrast in contrasts:
            epoch,kind,arm,rate=(contrast[k] for k in ('epoch','kind','arm','learning_rate'));terms=contrast_terms(kind,arm,rate)
            values=[];require(len(contrast['pairs'])==6,'Six paired records')
            for seed,row in zip(SEEDS,contrast['pairs']):
                expected_terms=[{'endpoint_id':indexed[seed,lr,a,epoch]['roster']['endpoint_id'],'sign':sign} for lr,a,sign in terms]
                value=float(100*sum((frac(indexed[seed,lr,a,epoch]['scores'][metric])*sign for lr,a,sign in terms),Fraction(0)))
                require(row['pair_seed']==seed and row['terms']==expected_terms and row['missing']==[],'Paired terms')
                require(row['stratum']==('historical' if seed<=303 else 'new'),'Pair stratum')
                close(row['value_pp'],value,'Exact-rational pair contrast');values.append(value);pair_count+=1
            verify_summary(contrast['all_six_pairs'],values,kind=='interaction',metric+' all six')
            verify_summary(contrast['historical_three_pairs'],values[:3],False,metric+' historical three')
            verify_summary(contrast['new_three_pairs'],values[3:],False,metric+' new three')
            is_primary=epoch==40 and kind=='interaction' and metric=='bdd_ten_class_AP50_95'
            require(contrast['role']==('primary' if is_primary else 'secondary'),'Prespecified primary role')
            if is_primary:primary.append(contrast)
            contrast_count+=1
    require(bundle.analysis('SUMMARY')['primary_interactions']==primary,'Primary summary identity')
    return {'endpoints':84,'raw_scores':84*18,'contrasts':contrast_count,'paired_contrasts':pair_count,'t975_df5_independent':t975_df5(),'primary_interactions':2}

def verify_metric_cells(bundle):
    records=bundle.analysis('METRIC_384_CELLS_AND_EFFECTS')
    roster=bundle.read(PHASE+'/factorial_endpoint_location_plan.json')['metric_decomposition_final_endpoint_ids']
    require(len(records)==48 and [r['endpoint_id'] for r in records]==roster,'48 fixed metric archives')
    checks=0
    for row in records:
        raw=bundle.read(row['evidence']['path'],row['evidence']['sha256'])
        require(raw['cells']==row['cells'] and raw['native_KITTI_equivalence_claimed'] is False,'Original metric result preserved')
        require(set(raw['common_reference_cells_exactly_reproduced'])=={'kitti_source_ignore','kitti_no_ignore_sensitivity'},'Two exact common reference cells')
        cells={((1 if c['overlap_policy']=='class_specific_iou' else 0),(1 if c['recall_points']==40 else 0),(1 if c['ignore_policy']=='no_ignore' else 0)):c for c in row['cells']}
        require(len(row['cells'])==8 and set(cells)==set(itertools.product((0,1),repeat=3)),'Eight metric cells')
        effects=row['factor_effects'];require(effects['independent_replications_claimed'] is False and effects['native_KITTI_equivalence_claimed'] is False,'Metric interpretation')
        names=('overlap','recall_grid','ignore');expected={tuple(names[i] for i in indices):indices for size in (1,2,3) for indices in itertools.combinations(range(3),size)}
        require(len(effects['effects'])==7 and {tuple(e['factors']) for e in effects['effects']}==set(expected),'Seven metric effects')
        for effect in effects['effects']:
            selected=expected[tuple(effect['factors'])];require(effect['order']==len(selected),'Interaction order')
            for name in ('mean_AP','person','car'):
                total=Fraction(0)
                for key,cell in cells.items():
                    x=cell['mean_AP'] if name=='mean_AP' else cell['class_AP'][name]
                    require(0<=x<=1,'Metric AP fraction')
                    sign=(-1)**sum(key[j]==0 for j in selected);total+=sign*frac(x)
                value=float(100*total/(2**(3-len(selected))))
                close(effect['effect_pp'][name],value,'Independent metric factorial contrast');checks+=1
    return {'archives':48,'cells':384,'reference_cells':96,'factor_effect_scalars':checks}

def verify_epoch_source(raw,summary,historical):
    literals=[]
    def constant(value):
        literals.append(value);return {'nonfinite_literal':value}
    def unique(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate raw epoch key');result[key]=value
        return result
    source=json.loads(raw,object_pairs_hook=unique,parse_constant=constant)
    median=source['median_unclipped_grad_norm'];reason=None
    if literals:
        require(historical and len(literals)==1 and median=={'nonfinite_literal':literals[0]},'Unexpected nonfinite raw epoch value')
        reason='historical_'+literals[0]+'_median_does_not_identify_nonfinite_backward_count'
        source['median_unclipped_grad_norm']=None
    # Strict round trip rejects overflow-to-infinity and any other nonfinite value.
    json.dumps(source,allow_nan=False)
    for key in ('epoch','lr','det_loss','feature_loss','batches','optimizer_steps_skipped','median_unclipped_grad_norm','first_two_input_sha256'):
        require(summary[key]==source[key],'Raw training epoch summary: '+key)
    for key in ('gradient_norm_nonfinite_batches','finite_gradient_clipped_batches'):
        require(summary[key]==source.get(key),'Raw training epoch diagnostic: '+key)
    require(summary['gradient_median_unavailable_reason']==reason,'Raw historical median reason')
    if not historical:
        for key in ('gradient_norm_finite_batches','gradient_norm_nonfinite_batches','finite_gradient_clipped_batches'):
            require(type(source[key]) is int and source[key]>=0,'Native gradient counter type')
        require(source['gradient_norm_finite_batches']+source['gradient_norm_nonfinite_batches']==source['batches'],'Native finite/nonfinite batch partition')
        require(source['finite_gradient_clipped_batches']<=source['gradient_norm_finite_batches'],'Native clipped batch bound')
        if source['median_unclipped_grad_norm'] is None:
            require(source['gradient_norm_nonfinite_batches']>0,'Native explicit null median requires recorded nonfinite batch')
    return source


def verify_metric_reference_parity(bundle):
    endpoints={r['roster']['endpoint_id']:r for r in bundle.analysis('ALL_84_ENDPOINTS')};count=0
    for row in bundle.analysis('METRIC_384_CELLS_AND_EFFECTS'):
        for policy,key in (('source_ignore','kitti_source_ignore'),('no_ignore','kitti_no_ignore')):
            cells=[c for c in row['cells'] if c['ignore_policy']==policy and c['overlap_policy']=='iou_grid' and c['recall_points']==101]
            require(len(cells)==1,'Single common reference metric cell')
            cell=cells[0];ref=endpoints[row['endpoint_id']]['evidence'][key]
            score=bundle.read(ref['artifact'],ref['sha256'])['score']
            require(cell['mean_AP']==score['mean_AP50_95'] and cell['class_AP']==score['class_AP50_95'],'Exact downloaded common reference parity')
            close(cell['mean_AP'],(cell['class_AP']['person']+cell['class_AP']['car'])/2,'Mean common reference AP')
            count+=1
    require(count==96,'All96 reference cells compared to original score bytes')
    return count


def verify_training(bundle):
    report=bundle.analysis('TRAINING_36_PATHS_AND6_NEW_BASES');rows=report['trajectories']
    require(len(rows)==36 and len({r['label'] for r in rows})==36,'36 training trajectories')
    require({(r['pair_seed'],r['learning_rate'],r['arm']) for r in rows}==set(itertools.product(SEEDS,RATES,ARMS)),'Training roster')
    reference={};comparisons=0;legacy_null=0;native_null=0;missingness=[]
    for row in rows:
        old=row['pair_seed'] in SEEDS[:3] and row['learning_rate']==RATES[0]
        require(row['origin']==('historical_reuse' if old else 'new'),'Training historical status')
        audit=bundle.read(row['audit']['path'],row['audit']['sha256'])
        if old:
            require(audit['status']=='passed' and audit['epochs']==40,'Raw historical training audit')
            counters={'applied':audit['optimizer_applied_updates'],'skipped':audit['optimizer_skipped_updates']}
            counters['attempted']=counters['applied']+counters['skipped']
            finite=audit['finite_final_student_adapter_ema_optimizer_tensors']
        else:
            require(audit['status']=='completed_evidence_passed','Raw new training audit')
            counters=audit['actual_step_counts'];finite=audit['finite_saved_states']
            require(row['observed_nonfinite_gradient_batches']==audit['native_gradient_nonfinite_batches'],'Raw audited nonfinite batch count')
        require(row['actual_step_counts']==counters and row['finite_saved_states']==finite,'Raw training audit counters and state')
        epochs=row['epochs'];require([e['epoch'] for e in epochs]==list(range(1,41)),'40 completed epochs')
        c=row['actual_step_counts'];require(c['attempted']==c['applied']+c['skipped']==43680,'Optimizer total identity')
        require(c['attempted']==sum(e['batches'] for e in epochs) and c['skipped']==sum(e['optimizer_steps_skipped'] for e in epochs),'Optimizer epoch sum')
        require(row['finite_saved_states'] is True,'Finite saved training state')
        if old:require(row['observed_nonfinite_gradient_batches'] is None and row['nonfinite_batch_count_unavailable_reason'],'Historical unavailable backward count')
        cumulative={'attempted':0,'applied':0,'skipped':0}
        for epoch in epochs:
            native=verify_epoch_source(bundle.raw(epoch['source']['path'],epoch['source']['sha256']),epoch,old)
            if not old:
                cumulative['attempted']+=epoch['batches'];cumulative['skipped']+=epoch['optimizer_steps_skipped']
                cumulative['applied']=cumulative['attempted']-cumulative['skipped']
                require(native['actual_step_counts']==cumulative,'Native cumulative optimizer counts')
            key=(row['pair_seed'],epoch['epoch']);digest=epoch['first_two_input_sha256'];require(len(digest)==2,'Two sampled input hashes')
            if key in reference:require(reference[key]==digest,'Independent sampled-input pairing');comparisons+=1
            else:reference[key]=digest
            if epoch['median_unclipped_grad_norm'] is None:
                if old:
                    require(epoch['gradient_median_unavailable_reason'],'Explicit historical median missingness');legacy_null+=1
                    reason=epoch['gradient_median_unavailable_reason']
                else:
                    require(epoch['gradient_median_unavailable_reason'] is None and epoch['gradient_norm_nonfinite_batches']>0,'Native explicit null retained with measured batch counter');native_null+=1
                    reason='native_explicit_null_median_with_recorded_nonfinite_gradient_batches'
                missingness.append(dict(label=row['label'],epoch=epoch['epoch'],origin=row['origin'],derived_explanation=reason,
                    observed_nonfinite_gradient_batches=epoch['gradient_norm_nonfinite_batches'],source=epoch['source']))
            else:require(math.isfinite(epoch['median_unclipped_grad_norm']),'Finite diagnostic median')
        if not old:require(sum(e['gradient_norm_nonfinite_batches'] for e in epochs)==row['observed_nonfinite_gradient_batches'],'Native nonfinite batch aggregate')
    require(comparisons==1200,'1200 sampled comparisons')
    require(len(report['new_bases'])==6,'Six new base diagnostics')
    for base in report['new_bases']:
        raw=bundle.read(base['audit']['path'],base['audit']['sha256'])['run']
        for key in ('model','seed','epochs','actual_step_counts','native_EMA_updates','initial_parameter_digest','raw_final_parameter_digest','EMA_parameter_digest','raw_changed_parameter_tensors','fixed_dfl_preserved','checkpoint_matches_raw_EMA_at_saved_precision','finite_saved_tensors'):
            require(base[key]==raw[key],'Raw base audit summary: '+key)
    for group,key in [(report['new_bases'],'new_base_optimizer_totals'),([r for r in rows if r['origin']=='new'],'new27_optimizer_totals'),([r for r in rows if r['origin']=='historical_reuse'],'historical9_optimizer_totals')]:
        require(report[key]=={k:sum(r['actual_step_counts'][k] for r in group) for k in ('attempted','applied','skipped')},'Training aggregate counters')
    policy=PHASE+'/continuation_evidence_checks.py';policy_sha='d2c88fff38b4611abe54c5d1b2d310663c3434b2913850c1936fc5dad464d4cf'
    bundle.raw(policy,policy_sha)
    return {'trajectories':36,'epochs':1440,'raw_epoch_sources_compared':1440,'raw_training_audits_compared':42,'sampled_pair_comparisons':comparisons,
        'historical_undefined_epoch_medians':legacy_null,'new_explicit_null_epoch_medians':native_null,
        'native_null_policy_evidence':{'path':policy,'sha256':policy_sha,'rule':'Native admission permits an explicit null median only with a positive measured nonfinite-gradient batch count. Source values and the legacy-only reason field are preserved unchanged.'},
        'unavailable_epoch_medians':missingness,'historical_nonfinite_backward_counts_reconstructed':False}

def verify_gradients(bundle):
    rows=bundle.analysis('GRADIENT_72_SUMMARY');require(len(rows)==72,'72 fixed gradient probes')
    require({(r['pair_seed'],r['learning_rate'],r['arm'],r['batch_index']) for r in rows}==set(itertools.product(SEEDS,RATES,ARMS,(0,1))),'Fixed gradient roster')
    status_counts={};undefined_groups=0;groups_checked=0
    for row in rows:
        raw=bundle.read(row['evidence']['path'],row['evidence']['sha256'])
        for key in ('label','pair_seed','learning_rate','arm','batch_index','status','detection_loss','feature_loss','groups'):require(row[key]==raw[key],'Gradient source summary: '+key)
        require(raw['teacher_no_grad'] and raw['model_state_unchanged'] and raw['input_tensors_unchanged'] and raw['parameter_grad_fields_unchanged'],'Gradient immutability flags')
        require(raw['batch_size']==32 and raw['feature_weight']==(0 if row['arm']=='control' else .3),'Fixed gradient batch/coefficient')
        status_counts[row['status']]=status_counts.get(row['status'],0)+1
        require(row['status'] in ('computed','nonfinite_objective','nonfinite_gradient'),'Gradient undefined outcome retained')
        for name,g in row['groups'].items():
            for kind in ('detection','feature','weighted_feature'):
                s=g[kind];nonfinite=sum(s[k] for k in ('nan_elements','positive_inf_elements','negative_inf_elements'))
                require((s['norm'] is None)==bool(nonfinite),'Gradient norm missingness')
                if s['norm'] is not None:require(math.isfinite(s['norm']) and s['norm']>=0,'Gradient norm validity')
            d,w=g['detection']['norm'],g['weighted_feature']['norm'];ratio=g['weighted_feature_to_detection_norm_ratio'];cosine=g['detection_weighted_feature_cosine'];dot=g['detection_weighted_feature_dot']
            if d is None or w is None:
                require(ratio is None and cosine is None and dot is None and 'nonfinite_detection_or_weighted_feature_gradient' in g['undefined_reasons'],'Nonfinite gradient ratio/cosine')
            else:
                if d>0:close(ratio,w/d,'Gradient norm ratio',tol=2e-10)
                else:require(ratio is None and 'zero_detection_gradient' in g['undefined_reasons'],'Zero detection denominator')
                if d>0 and w>0:
                    require(-1<=cosine<=1,'Cosine range');close(cosine,max(-1,min(1,dot/(d*w))),'Gradient cosine',tol=2e-10)
                else:require(cosine is None and g['undefined_reasons'],'Zero gradient cosine')
                if w==0:require('zero_weighted_feature_gradient' in g['undefined_reasons'],'Zero feature reason')
            if g['undefined_reasons']:undefined_groups+=1
            if row['arm']=='control':require(g['weighted_feature']['norm']==0 and row['feature_loss']==0,'Control feature gradient')
            groups_checked+=1
    return {'probes':72,'status_counts':status_counts,'groups_checked':groups_checked,'groups_with_undefined_values':undefined_groups,'gradient_tensors_recomputed':False}

def verify_execution_evidence(b):
    """Check actual exits, adoption, preserved failures and compatibility receipts."""
    expected='9ed35b5175b0bfdcc1c506f727f0164309e667fe04b73b5ab07d7ebb4442012f'
    require(b.index['source_manifest_sha256']==expected,'Admitted final compatibility manifest')
    bundles=[n.removesuffix('/SOURCE_MANIFEST.json') for n,d in b.checked.items() if n.endswith('/SOURCE_MANIFEST.json') and d==expected]
    require(len(bundles)==1,'One current admitted bundle')
    current=bundles[0];manifest=b.read(current+'/SOURCE_MANIFEST.json',expected)
    require(manifest['remote_folder']==current,'Admitted bundle identity')
    for name,digest in manifest['files'].items():require(b.checked.get(current+'/'+name)==digest,'Exported admitted source: '+name)
    controller=current+'/fixed_final_dispatch';completion=b.read(controller+'/COMPLETE.json')
    require(completion['status']=='all_fixed_scientific_stages_exited_and_independently_consumed' and set(completion['consumed'])=={'evaluation','gradient','metrics','analysis'},'All four final stages consumed')
    require(b.checked[controller+'/COMPLETE.json']==b.index['source_controller_complete_sha256'],'Export controller binding')
    require(b.read(controller+'/STARTED.json')['stages']==['metrics','analysis'],'Completed evaluation and gradient must not be dispatched again')
    retained=b.read(current+'/RETAINED_EVALUATION.json');old=retained['folder'].removesuffix('/evaluation_attempt01')
    require(retained['manifest_sha256']=='b0426bc08ef4a9a3c04ef0046f05740b891bdc4844d787bc5978e13ba822603f','Original evaluation manifest')
    old_manifest=b.read(old+'/SOURCE_MANIFEST.json',retained['manifest_sha256'])
    for name,digest in old_manifest['files'].items():require(b.checked.get(old+'/'+name)==digest,'Original evaluation source retained: '+name)
    adoption=b.read(controller+'/evaluation_ADOPTED.json',completion['consumed']['evaluation'])
    require(adoption['status']=='completed_evaluation_retained_without_reexecution' and adoption['new_inference'] is False and adoption['new_scoring'] is False,'Honest completed evaluation adoption')
    require(adoption['original_manifest_sha256']==retained['manifest_sha256'] and adoption['adopting_manifest_sha256']==expected,'Both evaluation/adoption manifests')
    for name,digest in retained['bindings'].items():require(b.checked.get(name)==digest,'Retained evaluation exit/seal binding')
    retained_gradient=b.read(current+'/RETAINED_GRADIENT_AND_CPU.json');gradient_bundle=retained_gradient['gradient_folder'].removesuffix('/gradient_attempt01')
    require(retained_gradient['manifest_sha256']=='56c7d1a641a14503e068c1aff79f6fda5293bda2d7e2eb66d9cc40905794816b','Original completed gradient manifest')
    gradient_manifest=b.read(gradient_bundle+'/SOURCE_MANIFEST.json',retained_gradient['manifest_sha256'])
    for name,digest in gradient_manifest['files'].items():require(b.checked.get(gradient_bundle+'/'+name)==digest,'Retained gradient and full CPU source: '+name)
    gradient_adoption=b.read(controller+'/gradient_ADOPTED.json',completion['consumed']['gradient'])
    require(gradient_adoption['status']=='completed_gradients_retained_without_reexecution' and gradient_adoption['new_probes'] is False,'Completed gradient adoption')
    require(gradient_adoption['original_manifest_sha256']==retained_gradient['manifest_sha256'] and gradient_adoption['adopting_manifest_sha256']==expected,'Gradient adoption manifests')
    for name,digest in retained_gradient['bindings'].items():require(b.checked.get(name)==digest,'Retained gradient and CPU bindings')
    stage_seals={}
    for key in ('CPU_integration','CPU_previous','evaluation','gradient','metrics','analysis'):
        stage='CPU_integration' if key=='CPU_previous' else key
        prefix=old if stage=='evaluation' else gradient_bundle if key in ('gradient','CPU_previous') else current
        seal_path=prefix+'/'+stage+'_attempt01/SEALED.json';seal=b.read(seal_path)
        require(seal['status']=='completed' and seal['stage']==stage and seal['terminal']['absent'] and not seal['terminal']['errors'],'Closed stage '+stage)
        stage_expected=retained['manifest_sha256'] if stage=='evaluation' else retained_gradient['manifest_sha256'] if key in ('gradient','CPU_previous') else expected
        require(seal['manifest_sha256']==stage_expected,'Stage source manifest '+stage)
        allowed=seal['result'].get('expected_failure_rows',{}) if stage=='CPU_integration' else {}
        for name,digest in seal['rows'].items():
            row=b.read(name,digest)
            require(row.get('ended_unix') and row['ended_unix']>=row['started_unix'],'Closed attempt clock')
            require(row['status']=='completed' and row['returncode']==0 or allowed.get(name)==digest,'Unexpected failed stage attempt')
        lease=b.read(seal['closed_lease'],seal['closed_lease_sha256'])
        require(lease['status']=='completed' and lease['all_registered_sessions_drained'],'Closed stage lease')
        if stage!='CPU_integration':
            stage_controller=prefix+'/fixed_final_dispatch'
            exit_record=b.read(stage_controller+'/'+stage+'_EXIT.json')
            proof_digest=retained['bindings'][retained['consumed_path']] if stage=='evaluation' else retained_gradient['bindings'][retained_gradient['gradient_consumed_path']] if stage=='gradient' else completion['consumed'][stage]
            proof=b.read(stage_controller+'/'+stage+'_CONSUMED.json',proof_digest)
            require(exit_record['returncode']==proof['returncode']==0 and exit_record['pid']==seal['owner']['pid'],'Actual successful child exit')
            require(proof['status']=='actual_exit_and_independent_full_consumption_passed' and proof['stage']==stage and proof['manifest_sha256']==seal['manifest_sha256'],'Independent stage consumption')
            require(proof['seal_sha256']==b.checked[seal_path],'Consumed seal bytes')
        stage_seals[key]=seal
    require(stage_seals['evaluation']['result']['new_endpoints']==60 and stage_seals['evaluation']['result']['reused_endpoints']==24,'Completed fixed evaluation count')
    recovery=b.read(current+'/GRADIENT_RECOVERY.json')
    require(b.analysis('SUMMARY')['gradient_serialization_compatibility']==recovery,'Compatibility disclosure in analysis')
    require(recovery['failed_research_probes']==recovery['failed_model_forwards']==0 and recovery['completed_evaluation_reexecuted'] is False,'Pre-forward failure scope')
    charge=0
    baseline=b.read(current+'/BASELINE.json')
    for name in recovery['failed_attempts']:
        row=b.read(name,baseline['baseline_attempts'][name]);require(row['status']=='failed' and row.get('ended_unix'),'Preserved failed fixture row')
        if row['GPU_job']:charge+=row['ended_unix']-row['started_unix']
    close(recovery['prior_gradient_GPU_reservation_seconds'],charge,'Failed fixture resource charge')
    close(recovery['remaining_gradient_GPU_hours'],6-charge/3600,'Original gradient allowance retained')
    corrected_charge=sum(b.read(n,d)['ended_unix']-b.read(n,d)['started_unix'] for n,d in stage_seals['gradient']['rows'].items() if b.read(n,d)['GPU_job'])
    require(charge+corrected_charge<=6*3600,'Combined failed and corrected gradient GPU budget')
    stopped=recovery['subsequent_CPU_admission_failure']
    require(stopped['checkpoint_loads']==0 and stopped['GPU_work'] is False and stopped['model_forwards']==0,'CPU timeout pre-model scope')
    for name in stopped['failed_rows']:
        row=b.read(name,baseline['baseline_attempts'][name]);require(row['status']=='failed' and row.get('ended_unix') and row['GPU_job'] is False,'Closed failed CPU attempt')
    original=PHASE+'/prepared_history/gradient_objectives_preparation_20261022_0535/gradient_probe_objectives.py'
    raw=b.raw(original,recovery['original_objectives_sha256']).decode('utf-8').replace('\r\n','\n')
    compatible=b.raw(current+'/gradient_objectives_compatible.py',recovery['corrected_objectives_sha256']).decode('utf-8').replace('\r\n','\n')
    require(compatible.replace('from collections import OrderedDict\n','').replace('type(state) in (dict, OrderedDict)','type(state) is dict')==raw,'Only container acceptance changes in objective source')
    cpu=stage_seals['CPU_previous']['result'];require(cpu['case_count']==len(cpu['cases'])==15 and len(set(cpu['cases']))==15,'Retained actual15 CPU cases')
    require(cpu['parent_model_loads']==cpu['model_forwards']==0 and cpu['GPU_work'] is False and len(cpu['CPU_affinity'])==4,'CPU scope and affinity')
    reference=cpu['actual36_compatibility_result'];compatibility=b.read(reference['path'],reference['sha256'])
    require(compatibility['status']=='all36_authenticated_CPU_assemblies_passed' and compatibility['paths']==len(compatibility['reports'])==36,'Actual36 CPU assemblies')
    require(compatibility['raw_state_loads']==36 and compatibility['model_template_loads']==72 and compatibility['model_forwards']==0 and compatibility['CUDA_initialized'] is False,'CPU checkpoint compatibility scope')
    require(set(compatibility['positive_container_cases'])=={'dict','OrderedDict'} and len(compatibility['rejected_cases'])==10,'Container regression cases')
    probes=b.analysis('GRADIENT_72_SUMMARY');labels={r['label'] for r in probes}
    require({r['label'] for r in compatibility['reports']}==labels and len(labels)==36,'CPU admission matches all research paths')
    for row in compatibility['reports']:
        require(row['exact_state_load_passed'] and row['model_forwards']==0 and all(c['metadata_preserved'] for c in row['containers'].values()),'Raw-state metadata preservation')
        for probe in (p for p in probes if p['label']==row['label']):
            actual=b.read(probe['evidence']['path'],probe['evidence']['sha256'])
            require(actual['raw_state']=={'path':row['raw_state'],'sha256':row['raw_state_sha256']},'Same admitted raw state in probe')
    fixture=b.read(recovery['corrected_result_directory']+'/synthetic_correctness/COMPLETE.json')
    require(fixture['status']=='passed' and fixture['synthetic_full32x640_cases']==3 and fixture['actual_study_batch_probes']==0 and fixture['saved_inputs_unchanged'],'Full-size synthetic GPU admission')
    require(set(fixture['reports'])==set(ARMS) and all(v['status']=='computed' and v['teacher_no_grad'] and v['model_state_unchanged'] for v in fixture['reports'].values()),'All three synthetic arm checks')
    targeted=stage_seals['CPU_integration'];target_result=targeted['result']
    require(target_result['case_count']==len(set(target_result['cases']))==5 and target_result['actual_native_publications']==66,'Targeted five filesystem cases')
    require(target_result['parent_model_loads']==target_result['model_forwards']==target_result['scientific_scorers_run']==0 and target_result['GPU_work'] is False,'Targeted filesystem scope')
    require(targeted['wall_seconds']<600 and targeted['CPU_seconds']<600 and target_result['old_published_bytes_preserved'],'Targeted filesystem caps and no replacement')
    reference=target_result['retained_previous_CPU'];require(reference['seal_sha256']==b.checked[retained_gradient['CPU_folder']+'/SEALED.json'] and reference['manifest_sha256']==retained_gradient['manifest_sha256'],'Targeted admission retains original full CPU proof')
    native_publications=[n for n in targeted['artifacts'] if n.endswith('.json') and ('/metric_publication_fixture/requests/' in n or '/analysis_publication_fixture/contrasts/' in n)]
    require(len(native_publications)==66,'All66 actual nested publications')
    for name in native_publications:require(b.read(name,targeted['artifacts'][name]['sha256'])['fixture_only'] is True,'Native publication remains synthetic')
    metric_recovery=b.read(current+'/METRICS_RECOVERY.json');require(b.analysis('SUMMARY')['metrics_filesystem_recovery']==metric_recovery,'Metrics filesystem disclosure')
    require(metric_recovery['metric_scorer_workers_spawned']==metric_recovery['metric_score_cells_created']==0 and metric_recovery['completed_results_reexecuted'] is False,'Pre-scorer failure scope')
    for name in metric_recovery['failed_metric_rows']:
        row=b.read(name,baseline['baseline_attempts'][name]);require(row['status']=='failed' and row.get('ended_unix') and row['GPU_job'] is False,'Preserved pre-scorer failure row')
    require(metric_recovery['actual_failed_CPU_seconds_available'] is False and metric_recovery['conservative_failed_CPU_seconds_reserved']==1800,'Unknown CPU retained with conservative charge')
    require(metric_recovery['failed_metrics_dispatch_to_exit_wall_seconds']*len(metric_recovery['available_CPU_affinity'])<1800,'Conservative failed CPU reservation bounds wall times CPUs')
    close(metric_recovery['remaining_metric_CPU_hours'],23.5-1800/3600,'Original metric CPU allowance retained')
    require(stage_seals['metrics']['CPU_seconds']<metric_recovery['remaining_metric_CPU_hours']*3600,'Final metric CPU allowance')
    return dict(actual_exit_and_consumption_stages=4,completed_evaluation_reexecuted=False,completed_gradients_reexecuted=False,native_CPU_cases_retained=15,targeted_filesystem_CPU_cases=5,
        authenticated_CPU_raw_state_assemblies=36,full32x640_synthetic_cases=3,original_objective_math_unchanged=True,
        prior_failed_gradient_GPU_seconds=charge,corrected_gradient_GPU_seconds=corrected_charge,prior_CPU_failure_preserved=True)


def run(root):
    b=Bundle(root);complete=b.analysis('COMPLETE')
    for name,digest in complete['bound_result_sha256'].items():require(b.checked[name]==digest,'Analysis completion hash')
    transition=b.analysis('SUMMARY')['environment_transition']
    input_proofs=[name for name in b.rows if name.endswith('/evaluation_attempt01/CURRENT_INPUT_BYTES.json')]
    require(len(input_proofs)==1 and b.read(input_proofs[0])['environment_transition']==transition,'Environment disclosure source')
    require(transition['historical_platform']=='Linux-6.17.0-40-generic-x86_64-with-glibc2.39' and transition['current_platform']=='Linux-7.0.0-38-generic-x86_64-with-glibc2.39','Reviewed kernel transition')
    require(transition['numerical_equivalence_due_to_kernel_change_established'] is False and transition['historical_results_recomputed'] is False,'Kernel limitation retained')
    result={'status':'independent_local_scalar_and_roster_verification_passed','exported_files_hash_checked':len(b.checked),
        'endpoint_and_contrast_checks':verify_endpoints_and_contrasts(b),'metric_checks':verify_metric_cells(b),
        'training_checks':verify_training(b),'gradient_checks':verify_gradients(b),'environment_transition':transition,
        'execution_and_compatibility_checks':verify_execution_evidence(b),
        'limits':['This local check verifies exported bytes, identities, arithmetic and summary consistency; it does not rerun model inference or native scorers.',
                  'Server independent audits bind model, image, label and prediction bytes; those large binary datasets are not downloaded here.',
                  'The historical48 strength-grid results remain a distinct study record;24 endpoints overlap the84 endpoint follow-up, giving108 distinct endpoints, not independent replications.']}
    historical=b.analysis('HISTORICAL_STRENGTH_GRID_REFERENCE');require(historical['endpoint_count']==48 and historical['scoring_repeated'] is False and historical['selection_performed'] is False,'Historical grid preserved')
    result['metric_checks']['downloaded_reference_score_cells_exactly_compared']=verify_metric_reference_parity(b)
    for key in ('plan','completion'):b.read(historical[key]['path'],historical[key]['sha256'])
    result['INDEX_sha256']=hashlib.sha256((Path(root)/'INDEX.json').read_bytes()).hexdigest()
    return result

if __name__=='__main__':
    root=Path(sys.argv[1]);result=run(root);destination=root/'INDEPENDENT_LOCAL_VERIFICATION.json'
    with destination.open('x',encoding='utf-8') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))
