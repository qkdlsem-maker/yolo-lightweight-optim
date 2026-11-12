"""Publication figures from independently verified final data only."""
import hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from verify_downloaded_results import Bundle,require
from prose_typography import readable

COLORS={101:'#31688E',202:'#26828E',303:'#35B779',404:'#8E3B46',505:'#AD6B22',606:'#67509A'}

def save(fig,root,name,caption,outputs):
    caption=caption+' Plotting code was prepared with OpenAI Codex assistance; coordinates come from the verified numerical records.'
    fig.savefig(root/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    outputs.append({'name':name,'caption':readable(caption),'png_sha256':hashlib.sha256((root/(name+'.png')).read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256((root/(name+'.pdf')).read_bytes()).hexdigest()});plt.close(fig)

def interaction(bundle,metric,arm):
    return next(c for c in bundle.analysis('contrasts/'+metric)['contrasts'] if c['kind']=='interaction' and c['epoch']==40 and c['arm']==arm)

def run(bundle_root,destination):
    source=Path(bundle_root);verified=json.loads((source/'INDEPENDENT_LOCAL_VERIFICATION.json').read_bytes())
    require(verified['status']=='independent_local_scalar_and_roster_verification_passed','Actual result verification required')
    require(verified['INDEX_sha256']==hashlib.sha256((source/'INDEX.json').read_bytes()).hexdigest(),'Changed verified index')
    bundle=Bundle(source);root=Path(destination);root.mkdir(parents=True,exist_ok=False);outputs=[]
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.transparent':False})
    primary='bdd_ten_class_AP50_95'
    fig,axes=plt.subplots(1,2,figsize=(7.3,4.2),layout='constrained',sharex=True,sharey=True)
    for ax,arm in zip(axes,('mse','cwd')):
        c=interaction(bundle,primary,arm);s=c['all_six_pairs'];ci=s['t_interval_95_pp']
        ax.axvline(0,color='#777777',lw=.8,zorder=0)
        for i,pair in enumerate(c['pairs']):
            seed=pair['pair_seed'];ax.scatter(pair['value_pp'],6-i,s=38,edgecolors=COLORS[seed],facecolors='white' if seed<=303 else COLORS[seed],zorder=3)
        ax.errorbar(s['mean_pp'],0,xerr=[[s['mean_pp']-ci['lower']],[ci['upper']-s['mean_pp']]],fmt='D',color='black',markersize=5,capsize=3,lw=1.2)
        ax.plot([ci['lower'],ci['upper']],[0,0],color='black',lw=1.2,zorder=2);ax.set_yticks([6,5,4,3,2,1,0],['101','202','303','404','505','606','Mean and 95% t interval'])
        ax.set_title(('A  ' if arm=='mse' else 'B  ')+arm.upper());ax.set_xlabel('Interaction (AP percentage points)');ax.set_ylim(-.65,6.65);ax.grid(axis='x',alpha=.16)
    save(fig,root,'Figure1_primary_interactions','Epoch40 native ten-class BDD AP50:95 interactions. Each circle is (KD minus control at0.001) minus (KD minus control at0.0001) within one base-training pair. Open circles are the three historical starting pairs and filled circles are the three new pairs. Diamonds show the mean and descriptive95% t interval with df5. Six matched pairs are the replication units; intervals do not provide multiplicity-adjusted significance or equivalence decisions.',outputs)
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.8),layout='constrained',sharey=True)
    contrasts=bundle.analysis('contrasts/'+primary)['contrasts']
    for ax,arm in zip(axes,('mse','cwd')):
        rates=[next(c for c in contrasts if c['kind']=='kd_minus_control' and c['epoch']==40 and c['arm']==arm and c['learning_rate']==lr) for lr in (.0001,.001)]
        ax.axhline(0,color='#777777',lw=.8)
        for i,seed in enumerate((101,202,303,404,505,606)):
            ax.plot([0,1],[c['pairs'][i]['value_pp'] for c in rates],marker='o',markersize=4,color=COLORS[seed],
                markerfacecolor='white' if seed<=303 else COLORS[seed],lw=1,label=str(seed))
        ax.set_xticks([0,1],['0.0001','0.001']);ax.set_xlim(-.18,1.18);ax.set_xlabel('Initial continuation learning rate');ax.set_title(('A  ' if arm=='mse' else 'B  ')+arm.upper());ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('KD minus control in AP percentage points');axes[1].legend(title='Pair seed',loc='best',fontsize=8,title_fontsize=8,frameon=False,ncol=2)
    save(fig,root,'Figure2_simple_effects','Epoch40 native BDD paired KD-minus-control differences at both fixed learning rates. Each line connects the same starting pair; the slope corresponds to the interaction shown in Figure1. The coefficient is0.3 in both distillation arms. These are paired observations, not 12 independent replications per method.',outputs)
    metrics=[('bdd_ten_class_AP50_95','BDD native\nten classes'),('bdd_common_AP50_95','BDD common\ntwo classes'),('kitti_common_source_ignore_AP50_95','KITTI common\nsource ignore'),('kitti_native_mean_moderate_R40','KITTI native\nmoderate R40')]
    fig,axes=plt.subplots(1,2,figsize=(7.3,4.1),layout='constrained',sharex=True,sharey=True)
    for ax,arm in zip(axes,('mse','cwd')):
        ax.axvline(0,color='#777777',lw=.8)
        for y,(metric,label) in enumerate(metrics):
            c=interaction(bundle,metric,arm)
            for j,p in enumerate(c['pairs']):ax.scatter(p['value_pp'],3-y+(j-2.5)*.045,s=16,color=COLORS[p['pair_seed']],marker='o' if p['pair_seed']<=303 else 's',alpha=.8)
            ax.scatter(c['all_six_pairs']['mean_pp'],3-y,s=48,color='black',marker='|',zorder=4)
        ax.set_yticks(range(4),[m[1] for m in metrics[::-1]]);ax.set_title(('A  ' if arm=='mse' else 'B  ')+arm.upper());ax.set_xlabel('Epoch 40 interaction (AP percentage points)');ax.grid(axis='x',alpha=.16)
    save(fig,root,'Figure3_evaluation_definitions','Pair-level epoch40 learning-rate interactions under four fixed evaluation definitions. Black ticks are six-pair means; small vertical offsets separate plotted pairs and have no numerical meaning. The full18-metric results also retain the no-ignore sensitivity and native KITTI class and difficulty outputs. Different class populations and matching rules limit direct comparison and do not identify a causal domain-shift mechanism.',outputs)
    gradients=bundle.analysis('GRADIENT_72_SUMMARY');fig,axes=plt.subplots(1,2,figsize=(7.3,4.1),layout='constrained',sharex=True,sharey=True);plotted=0;unplotted={}
    observed_cosines=[r['groups']['all_student']['detection_weighted_feature_cosine'] for r in gradients if r['arm'] in ('mse','cwd') and r['groups']['all_student']['detection_weighted_feature_cosine'] is not None]
    cosine_limit=min(1.03,max(.25,1.15*max((abs(v) for v in observed_cosines),default=1.0)))
    observed_ratios=[r['groups']['all_student']['weighted_feature_to_detection_norm_ratio'] for r in gradients if r['arm'] in ('mse','cwd') and r['groups']['all_student']['weighted_feature_to_detection_norm_ratio'] is not None and r['groups']['all_student']['weighted_feature_to_detection_norm_ratio']>0]
    for ax,arm in zip(axes,('mse','cwd')):
        for row in gradients:
            if row['arm']!=arm:continue
            g=row['groups']['all_student'];ratio=g['weighted_feature_to_detection_norm_ratio'];cosine=g['detection_weighted_feature_cosine']
            if ratio is None or cosine is None or ratio<=0:
                key=row['status']+':'+','.join(g['undefined_reasons']);unplotted[key]=unplotted.get(key,0)+1;continue
            ax.scatter(ratio,cosine,s=29,color=COLORS[row['pair_seed']],marker='o' if row['learning_rate']==.0001 else '^',
                facecolors='none' if row['batch_index']==0 else COLORS[row['pair_seed']],linewidths=.9);plotted+=1
        ax.axhline(0,color='#777777',lw=.8);ax.set_xscale('log');ax.set_ylim(-cosine_limit,cosine_limit);ax.set_title(('A  ' if arm=='mse' else 'B  ')+arm.upper());ax.set_xlabel('Weighted feature / detection\ngradient norm ratio');ax.grid(alpha=.15)
        if observed_ratios:ax.set_xlim(min(observed_ratios)/1.15,max(observed_ratios)*1.15)
        ax.set_xticks([.02,.04,.08,.16],['0.02','0.04','0.08','0.16']);ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('Detection and weighted feature gradient cosine')
    caption=f'Fixed-model FP32 diagnostics on two prespecified32-image training batches per path. All-student parameter groups are shown on a shared logarithmic horizontal scale. Colors identify pair seeds as in Figure2; circles and triangles identify rates0.0001 and0.001; open and filled symbols identify batches0 and1. Plotted coordinates: {plotted} of48 KD batch probes. Unplotted KD coordinates: {sum(unplotted.values())}, retained with reasons in JSON. All24 control probes remain in the report; their zero feature gradients make cosine undefined. Raw final models, rather than EMA inference checkpoints, are measured. These probes do not reconstruct historical stochastic AMP updates or establish a causal mechanism.'
    save(fig,root,'Figure4_fixed_gradient_diagnostics',caption,outputs)
    factors=bundle.analysis('METRIC_384_CELLS_AND_EFFECTS');endpoints={e['roster']['endpoint_id']:e['roster'] for e in bundle.analysis('ALL_84_ENDPOINTS')}
    matrix=np.array([[x['effect_pp']['mean_AP'] for x in row['factor_effects']['effects']] for row in factors]);limit=max(float(np.abs(matrix).max()),1e-8)
    labels=[]
    for row in factors:
        r=endpoints[row['endpoint_id']]
        if r['stage']=='base':label=f"{r['pair_seed']} {r['architecture']} base"
        else:label=f"{r['pair_seed']} {r['arm']} {'low' if r['lr']==.0001 else 'high'}"
        labels.append(label)
    fig,axes=plt.subplots(1,2,figsize=(7.0,8.0),layout='constrained')
    for i,ax in enumerate(axes):
        part=matrix[i*24:(i+1)*24];im=ax.imshow(part,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_yticks(range(24),labels[i*24:(i+1)*24],fontsize=7.5)
        ax.set_xticks(range(7),['O','R','I','O R','O I','R I','O R I'],fontsize=8);ax.set_title('Final endpoints '+str(i*24+1)+' to '+str((i+1)*24))
    fig.colorbar(im,ax=axes,label='Mean two-class AP effect in percentage points',shrink=.7,pad=.02)
    save(fig,root,'FigureS1_all_metric_factor_effects','All48 fixed final archives and all seven controlled metric contrasts for equally weighted person/car AP. O: class-specific overlap thresholds minus the IoU grid; R: 40-point minus101-point recall integration; I: no-ignore minus source-ignore. Main effects average over the remaining factors; interactions are differences of differences. Each archive supplies eight scores from unchanged predictions. These are within-archive sensitivity measures, not 48 independent model replications, and do not reproduce native KITTI scoring. Full class-specific contrasts and384 cells accompany the report.',outputs)
    value={'status':'figures_built_from_verified_results','INDEX_sha256':verified['INDEX_sha256'],'figures':outputs,
        'builder_sources':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('build_result_figures.py','prose_typography.py','verify_downloaded_results.py')},
        'gradient_coordinates_plotted':plotted,'unplotted_KD_coordinates_by_reason':unplotted,'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__}
    (root/'FIGURE_MANIFEST.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');return value

if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
