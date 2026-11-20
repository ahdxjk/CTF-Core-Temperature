"""Small, frozen, validation-only four-family fusion exploration. CPU only."""
from pathlib import Path
import os, sys, json, hashlib, time, argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import psutil
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from build_five_phases_v1 import label_trial, synthetic_checks
PARENT=ROOT/'outputs/ctf_direct_forecast_20261113'
H10=ROOT/'outputs/ctf_direct_forecast_h10_20261114'
PLANS=ROOT/'outputs/ctf_activity_plan_20261114'
FAMILIES=['HE','MSCA','PFN','ET']
HORIZONS=[5,10,15,30]
CONDITIONS=['CurrentOnly','SimplePlan']
PHASES=['P1','P2','P3','P4','P5']
SEEDS=[20261103,20261104]
SPLITS=json.loads((ROOT/'outputs/cpu_baselines_v1/splits.json').read_text(encoding='utf8'))
CANDIDATES=['Equal','Global_r010','Global_r002','Horizon_r010','Horizon_r002','Phase_r050','Phase_r010']
EQ=np.full(4,.25)
META=['source_excel_row','Participant','Condition','trial','minute','phase_5','target_excel_row','truth','fold','horizon']
COUNTS={5:16389,10:15619,15:14845,30:12534}
EPS=.001
def log(*v):print(time.strftime('%Y-%m-%d %H:%M:%S'),*v,flush=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def sources(condition,h,fold,role):
    folder=PLANS/'SimplePlan' if condition=='SimplePlan' else H10 if h==10 else PARENT
    result={}
    for kind in FAMILIES:
        tags=[f'{kind}_h{h}_s{seed}_f{fold}' for seed in SEEDS] if kind in ['HE','MSCA'] else [f'{kind}_h{h}_f{fold}']
        result[kind]=[(folder/'predictions'/f'{tag}_{role}.csv.gz',folder/'records'/f'{tag}.json',folder/'protocol.json') for tag in tags]
    return result
def load_matrix(condition,h,split,role):
    q=None
    for kind,paths in sources(condition,h,split['fold'],role).items():
        preds=[]
        for path,record,protocol in paths:
            rec=read(record);assert rec['protocol_sha']==sha(protocol)
            assert rec['hashes'][str(path)]==sha(path)
            a=pd.read_csv(path).sort_values('source_excel_row').reset_index(drop=True)
            assert set(a.Participant)==set(split[role])
            assert a.source_excel_row.is_unique and np.isfinite(a.prediction).all()
            if q is None:q=a[META].copy()
            for c in META:np.testing.assert_array_equal(q[c],a[c])
            preds.append(a.prediction.to_numpy())
        q[kind]=np.mean(preds,axis=0)
    return q
def phase_audit():
    d=pd.read_pickle(ROOT/'outputs/strict_confirmation_v1/clean_raw.pkl')
    cols=['trial','minute','workrest'];raw=d[cols]
    parts=[];prefixes=0
    synthetic_checks()
    for _,g in raw.groupby('trial',sort=False):
        v=label_trial(g);parts.append(v)
        for length in sorted(set([1,min(12,len(g)),len(g)//2,len(g)])):
            np.testing.assert_array_equal(label_trial(g.iloc[:length]).phase_5,v.phase_5.iloc[:length]);prefixes+=1
        altered=g.copy();cut=len(g)//2;altered.loc[altered.index[cut:],'workrest']=1
        np.testing.assert_array_equal(label_trial(altered).phase_5.iloc[:cut],v.phase_5.iloc[:cut])
    phases=pd.concat(parts).sort_index().phase_5
    np.testing.assert_array_equal(phases,d.phase_5)
    return dict(status='passed',rows=len(d),trials=d.trial.nunique(),prefix_checks=prefixes,future_state_perturbation=True,
        temperature_columns_never_passed=True,labels_match_archives=True,phase_counts=phases.value_counts().to_dict())
def prepare():
    deps={}
    paths=[Path(__file__),ROOT/'build_five_phases_v1.py',ROOT/'outputs/strict_confirmation_v1/clean_raw.pkl',ROOT/'outputs/cpu_baselines_v1/splits.json']
    for condition in CONDITIONS:
        for h in HORIZONS:
            for s in SPLITS:
                for role in ['validation','test']:
                    for vals in sources(condition,h,s['fold'],role).values():
                        for row in vals:paths.extend(row)
    for path in sorted(set(paths)):deps[str(path)]=sha(path)
    protocol=dict(version='frozen-four-family-forecast-fusion-20261115-v1',created=time.strftime('%Y-%m-%d %H:%M:%S'),
        dependency_hashes=deps,families=FAMILIES,seeds=SEEDS,conditions=CONDITIONS,horizons=HORIZONS,splits=SPLITS,candidates=CANDIDATES,
        optimization=dict(loss='participant macro smoothed absolute error sqrt(error^2+0.001^2); average horizons equally',
            solver='SLSQP with analytic gradient, simplex sum=1, per-member lower bound0.05 upper0.85, ftol1e-11 maxiter200',
            penalty='lambda * squared Euclidean distance from anchor, in degrees C',
            global_lambdas=[.10,.02],horizon_lambdas=[.10,.02],phase_lambdas=[.50,.10],
            global_anchor='equal weights',horizon_anchor='global fitted at lambda0.10',phase_anchor='horizon fitted at lambda0.10',
            phase_support='at least4 distinct fitting participants; else use horizon anchor',unknown_phase='horizon anchor'),
        selection='Per outer fold/condition: eight leave-one-validation-participant-out folds. Each held participant averages MAE across four horizons. MinimumCV mean plus its between-person SE defines one-SE threshold; choose first candidate in fixed simple-to-complex order within threshold. Refit all candidates on all8 validation participants, freeze selected candidate before loading test prediction values. Selected scheme common across horizons; Horizon/Phase weights remain horizon-specific.',
        phase='Causal activity state at forecast origin: P1 initial observed rest from minute0; P2/P4 first5min after observed start/stop; P3/P5 thereafter; unknown if state onset not observed/gaps. No target temperatures/future activities used.',
        limitation='Neural checkpoints already selected on these8 validation subjects and PFN mean/median chosen on them; ET settings inherited prior validation. Meta-level LOPO is NOT independently cross-fitted base predictions and does not remove prior validation-use bias. Test partitions already repeatedly analyzed. All results exploratory.',
        input_comparison='Phase fusion additionally uses causal activity history for CurrentOnly; retain History CTF reference. SimplePlan already includes observed activity history. No CatBoost or new experts.',
        prior_work='Earlier estimation BlendVal already convex weights, phase-calibration correction and learned gates exist. This experiment is a controlled forecast comparison with tiny fixed candidate set and meta-level participant CV, not proof of algorithmic novelty.',
        no_gating_network='No free learned disagreement gating in this pass: eight validation people and phase weighting already add substantial capacity.',
        statistics='40-person macro MAE/RMSE/Bias and10000 paired participant bootstrap seed20261115; unadjusted exploratory; per-phase conditional participant means.',
        runtime='CPU2 threads only, no torch/GPU, no base-model training')
    if (OUT/'protocol.json').exists():
        prior=read(OUT/'protocol.json');protocol['created']=prior['created'];assert prior==protocol
    else:save(OUT/'protocol.json',protocol)
    save(OUT/'phase_audit.json',phase_audit())
    # Validation data only are opened before fitting and scheme selection.
    checks=[]
    for c in CONDITIONS:
        for s in SPLITS:
            assert len(set(s['train']+s['validation']+s['test']))==40
            for h in HORIZONS:
                q=load_matrix(c,h,s,'validation');checks.append(dict(condition=c,fold=s['fold'],horizon=h,rows=len(q),people=q.Participant.nunique()))
    save(OUT/'preflight.json',dict(status='passed',validation_checks=checks,test_values_loaded=False))
    log('PROTOCOL FROZEN',sha(OUT/'protocol.json'))
def check():
    p=read(OUT/'protocol.json')
    for file,digest in p['dependency_hashes'].items():assert sha(file)==digest,file
    assert read(OUT/'preflight.json')['status']=='passed'
def pw(q):
    return 1/(q.Participant.nunique()*q.Participant.map(q.Participant.value_counts()).to_numpy())
def optimize(datasets,anchor,lam):
    xx=np.concatenate([q[FAMILIES].to_numpy() for q in datasets]);yy=np.concatenate([q.truth.to_numpy() for q in datasets]);ww=np.concatenate([pw(q)/len(datasets) for q in datasets])
    # Centering avoids subtracting large near-equal temperatures in gradient.
    xx=xx-xx.mean(axis=1,keepdims=True);yy=yy-np.concatenate([q[FAMILIES].mean(axis=1).to_numpy() for q in datasets])
    def objective(w):
        e=xx@w-yy;r=np.sqrt(e*e+EPS*EPS);diff=w-anchor
        return float(ww@r+lam*(diff@diff)),xx.T@(ww*e/r)+2*lam*diff
    fit=minimize(objective,anchor.copy(),jac=True,method='SLSQP',bounds=[(.05,.85)]*4,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(4))],options=dict(ftol=1e-11,maxiter=200))
    assert fit.success,fit.message
    assert abs(fit.x.sum()-1)<1e-9 and np.min(fit.x)>=.05-1e-9
    return fit.x
def fit_candidates(data):
    out={'Equal':{'global':EQ.tolist(),'horizon':{},'phase':{}}}
    for suffix,lam in [('r010',.10),('r002',.02)]:
        out['Global_'+suffix]={'global':optimize(list(data.values()),EQ,lam).tolist(),'horizon':{},'phase':{}}
    anchor=np.array(out['Global_r010']['global'])
    for suffix,lam in [('r010',.10),('r002',.02)]:
        out['Horizon_'+suffix]={'global':anchor.tolist(),'horizon':{str(h):optimize([q],anchor,lam).tolist() for h,q in data.items()},'phase':{}}
    ha=out['Horizon_r010']['horizon']
    for suffix,lam in [('r050',.50),('r010',.10)]:
        ph={};supports={}
        for h,q in data.items():
            ph[str(h)]={};supports[str(h)]={}
            for phase in PHASES:
                sub=q[q.phase_5.eq(phase)];n=sub.Participant.nunique();supports[str(h)][phase]=int(n)
                a=np.array(ha[str(h)])
                ph[str(h)][phase]=(optimize([sub],a,lam) if n>=4 else a).tolist()
        out['Phase_'+suffix]={'global':anchor.tolist(),'horizon':ha,'phase':ph,'phase_fit_people':supports}
    return out
def predict(q,h,model):
    w=np.tile(np.array(model['horizon'].get(str(h),model['global'])),(len(q),1))
    for phase,weights in model['phase'].get(str(h),{}).items():w[q.phase_5.eq(phase)]=weights
    assert np.allclose(w.sum(axis=1),1) and w.min()>=.05-1e-9
    return (q[FAMILIES].to_numpy()*w).sum(axis=1)
def score_person(q,pred):return pd.Series(abs(pred-q.truth.to_numpy()),index=q.Participant).groupby(level=0).mean()
def select_fit():
    check()
    for condition in CONDITIONS:
        for split in SPLITS:
            folder=OUT/'fits'/f'{condition}_f{split["fold"]}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'selection.json').exists():
                rec=read(folder/'selection.json');assert rec['protocol_sha']==sha(OUT/'protocol.json')
                assert sha(folder/'models.json')==rec['models_sha'];continue
            data={h:load_matrix(condition,h,split,'validation') for h in HORIZONS};cv=[]
            for person in split['validation']:
                tr={h:q[~q.Participant.eq(person)] for h,q in data.items()};va={h:q[q.Participant.eq(person)] for h,q in data.items()}
                models=fit_candidates(tr)
                for name,model in models.items():
                    for h,q in va.items():cv.append(dict(Participant=person,method=name,horizon=h,MAE=score_person(q,predict(q,h,model)).iloc[0]))
            cv=pd.DataFrame(cv);cv.to_csv(folder/'cv_predictions_score.csv',index=False)
            person=cv.groupby(['method','Participant']).MAE.mean().unstack('method').reindex(columns=CANDIDATES)
            means=person.mean();ses=person.std(ddof=1)/np.sqrt(8);best=means.idxmin();cutoff=float(means[best]+ses[best])
            selected=next(name for name in CANDIDATES if means[name]<=cutoff+1e-12)
            models=fit_candidates(data);save(folder/'models.json',models)
            save(folder/'selection.json',dict(protocol_sha=sha(OUT/'protocol.json'),condition=condition,fold=split['fold'],
                validation_people=split['validation'],cv_mean=means.to_dict(),cv_se=ses.to_dict(),best_cv=best,one_se_threshold=cutoff,selected=selected,
                models_sha=sha(folder/'models.json'),cv_sha=sha(folder/'cv_predictions_score.csv'),test_values_loaded=False))
            log('SELECTED',condition,split['fold'],selected,'CV minimum',best,round(means[best],6))
    save(OUT/'selection_complete.json',dict(status='passed',protocol_sha=sha(OUT/'protocol.json'),groups=10,test_values_loaded=False))
def bootstrap(delta):
    d=np.asarray(delta);r=np.random.default_rng(20261115)
    return np.quantile(d[r.integers(len(d),size=(10000,len(d)))].mean(axis=1),[.025,.975])
def report():
    check();assert read(OUT/'selection_complete.json')['status']=='passed'
    frames=[];weights=[];sels=[]
    for c in CONDITIONS:
        for s in SPLITS:
            folder=OUT/'fits'/f'{c}_f{s["fold"]}';selection=read(folder/'selection.json');models=read(folder/'models.json')
            assert selection['models_sha']==sha(folder/'models.json');sels.append(selection)
            for name,model in models.items():
                for h in HORIZONS:
                    for phase in ['global',*PHASES]:
                        w=model['phase'].get(str(h),{}).get(phase,model['horizon'].get(str(h),model['global']))
                        weights.append(dict(condition=c,fold=s['fold'],method=name,horizon=h,phase=phase,**dict(zip(FAMILIES,w))))
            for h in HORIZONS:
                q=load_matrix(c,h,s,'test')
                for name,model in models.items():
                    a=q[META].copy();a['prediction']=predict(q,h,model);a['method']=name;a['condition']=c;frames.append(a)
                a=q[META].copy();a['prediction']=predict(q,h,models[selection['selected']]);a['method']='ValSelected';a['condition']=c;frames.append(a)
    q=pd.concat(frames,ignore_index=True)
    assert not q.duplicated(['condition','horizon','method','source_excel_row']).any()
    q['error']=q.prediction-q.truth;q['ae']=q.error.abs();q['se']=q.error**2
    keys=['condition','horizon','method'];p=q.groupby(keys+['Participant'],as_index=False).agg(MAE=('ae','mean'),MSE=('se','mean'),Bias=('error','mean'),origins=('ae','size'));p['RMSE']=np.sqrt(p.MSE)
    summary=p.groupby(keys,as_index=False).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),Bias=('Bias','mean'),people=('Participant','size'),origins=('origins','sum'))
    assert summary.people.eq(40).all()
    for h,g in summary.groupby('horizon'):assert g.origins.eq(COUNTS[h]).all()
    reference=pd.read_csv(PLANS/'CTF_main_table.csv')
    original=pd.read_csv(ROOT/'outputs/forecast_peer_extension_20261114/comparison_summary.csv')
    for r in summary[summary.method.eq('Equal')].itertuples():
        old=original[(original.condition==r.condition)&(original.horizon==r.horizon)&(original.method=='CTF')].MAE.item()
        assert abs(r.MAE-old)<1e-12
    paired=[]
    for (c,h),g in p.groupby(['condition','horizon']):
        wide=g.pivot(index='Participant',columns='method',values='MAE')
        for name in CANDIDATES+['ValSelected']:
            delta=wide[name]-wide.Equal;lo,hi=bootstrap(delta)
            paired.append(dict(condition=c,horizon=h,method=name,delta_vs_Equal=delta.mean(),low=lo,high=hi,wins=int((delta<-1e-12).sum()),people=len(delta)))
    phasep=q.groupby(keys+['phase_5','Participant'],as_index=False).agg(MAE=('ae','mean'),Bias=('error','mean'),origins=('ae','size'))
    phases=phasep.groupby(keys+['phase_5'],as_index=False).agg(MAE=('MAE','mean'),Bias=('Bias','mean'),people=('Participant','size'),origins=('origins','sum'))
    phaseci=[]
    for (c,h,phase),g in phasep.groupby(['condition','horizon','phase_5']):
        wide=g.pivot(index='Participant',columns='method',values='MAE')
        for name in ['ValSelected','Phase_r050','Phase_r010']:
            delta=wide[name]-wide.Equal;lo,hi=bootstrap(delta)
            phaseci.append(dict(condition=c,horizon=h,phase_5=phase,method=name,delta_vs_Equal=delta.mean(),low=lo,high=hi,people=len(delta)))
    q.to_csv(OUT/'predictions_all.csv.gz',index=False);p.to_csv(OUT/'participant.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_ci.csv',index=False);phases.to_csv(OUT/'phase_summary.csv',index=False);phasep.to_csv(OUT/'phase_participant.csv',index=False)
    pd.DataFrame(phaseci).to_csv(OUT/'phase_paired_ci.csv',index=False);pd.DataFrame(weights).to_csv(OUT/'weights.csv',index=False);save(OUT/'selections.json',sels)
    original[(original.method=='CTF')&(original.condition=='History')].to_csv(OUT/'History_CTF_reference.csv',index=False)
    save(OUT/'verification.json',dict(status='passed',protocol_sha=sha(OUT/'protocol.json'),no_base_training=True,
        candidates_frozen_before_fit=True,selection_frozen_before_test_loading=True,all_weights_simplex=True,all_origins_matched=True,equal_replays_original=True,
        limitation='Meta-level LOPO only. Previously validation-selected base models and repeatedly explored test folds; not independent nested OOF confirmation.',
        output_hashes={str(OUT/name):sha(OUT/name) for name in ['summary.csv','participant.csv','predictions_all.csv.gz','weights.csv','selections.json']}))
    print(summary.to_string(index=False));log('COMPLETE')
if __name__=='__main__':
    cpus=psutil.Process().cpu_affinity();psutil.Process().cpu_affinity(cpus[8:10] if len(cpus)>=10 else cpus[-2:]);threadpool_limits(2)
    for name in ['fits']:(OUT/name).mkdir(parents=True,exist_ok=True)
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['prepare','all']:prepare()
    if mode in ['fit','all']:select_fit()
    if mode in ['report','all']:report()
