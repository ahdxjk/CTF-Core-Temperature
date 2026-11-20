"""Frozen extension to current-temperature estimation. No base-model fitting."""
from pathlib import Path
import sys, time
import run_forecast_fusion as f
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import psutil
OUT=Path(__file__).resolve().parent/'estimation'
SOURCE=f.ROOT/'outputs/revision_round3_20261108/estimation'
PANELS=['Neck14','Base12','Neck2']
CANDIDATES=['Equal','Global_r010','Global_r002','Phase_r050','Phase_r010']
META=['source_excel_row','Participant','Condition','trial','minute','phase_5','Corerectal','panel']
def paths(panel,split,role):
    out={}
    for family in f.FAMILIES:
        tags=[f'{family}_{"Shared7" if family=="MSCA" else panel}_s{seed}_f{split["fold"]}' for seed in f.SEEDS] if family in ['HE','MSCA'] else [f'{family}_{panel}_f{split["fold"]}']
        out[family]=[(SOURCE/'predictions'/f'{tag}_{role}.csv.gz',SOURCE/'records'/f'{tag}.json') for tag in tags]
    return out
def load(panel,split,role):
    q=None
    for family,files in paths(panel,split,role).items():
        arr=[]
        for path,record in files:
            r=f.read(record);assert r['protocol_sha']==f.sha(SOURCE/'protocol.json')
            assert r['hashes'][str(path)]==f.sha(path)
            a=pd.read_csv(path);a=a[a.panel.eq(panel)].sort_values('source_excel_row').reset_index(drop=True)
            assert a.source_excel_row.is_unique and set(a.Participant)==set(split[role]) and np.isfinite(a.prediction).all()
            if q is None:q=a[META].copy()
            for c in META:np.testing.assert_array_equal(q[c],a[c])
            arr.append(a.prediction.to_numpy())
        q[family]=np.mean(arr,axis=0)
    q=q.rename(columns={'Corerectal':'truth'});q['fold']=split['fold']
    return q
def fit(q):
    models={'Equal':{'global':f.EQ.tolist(),'horizon':{},'phase':{}}}
    for suffix,lam in [('r010',.10),('r002',.02)]:models['Global_'+suffix]={'global':f.optimize([q],f.EQ,lam).tolist(),'horizon':{},'phase':{}}
    anchor=np.array(models['Global_r010']['global'])
    for suffix,lam in [('r050',.50),('r010',.10)]:
        phase={};support={}
        for ph in f.PHASES:
            a=q[q.phase_5.eq(ph)];n=a.Participant.nunique();support[ph]=int(n)
            phase[ph]=(f.optimize([a],anchor,lam) if n>=4 else anchor).tolist()
        models['Phase_'+suffix]={'global':anchor.tolist(),'horizon':{},'phase':{'0':phase},'phase_fit_people':support}
    return models
def prepare():
    files=[Path(__file__),Path(f.__file__),SOURCE/'protocol.json',f.ROOT/'outputs/strict_confirmation_v1/clean_raw.pkl']
    for panel in PANELS:
        for s in f.SPLITS:
            for role in ['validation','test']:
                for ff in paths(panel,s,role).values():
                    for pair in ff:files.extend(pair)
    protocol=dict(version='frozen-four-family-estimation-fusion-20261115-v1',created=time.strftime('%Y-%m-%d %H:%M:%S'),
        source_hashes={str(p):f.sha(p) for p in sorted(set(files))},panels=PANELS,primary='Neck14',families=f.FAMILIES,candidates=CANDIDATES,splits=f.SPLITS,
        rule='Identical frozen forecast optimization/CV policy but only current-temperature target; no horizon weights. Global lambdas0.10/0.02, phase lambdas0.50/0.10 anchored to Global0.10; min4 phase-fitting people else anchor; simplex minimum0.05 each.',
        selection='8-person validation LOPO; one-SE threshold from best candidate mean and between-person SE; first eligible in fixed simple-to-complex candidate order; full8person refit. All selections frozen before reading any test prediction values.',
        limitation='Neural checkpoints and prior ET/PFN configuration already validation-selected; meta-level LOPO does not repair prior validation-use bias. Same repeatedly explored test partitions. This is exploration, not fully nested independent OOF.',
        phase_information='Phase uses causal work/rest history and trial-start status not supplied to original estimators; any improvement may combine information gain and fusion gain. Unknown phases fall back to global.',
        no_training='Only4-dimensional convex weight optimization CPU2; no base-model or GPU fitting; no CatBoost.')
    if (OUT/'protocol.json').exists():
        protocol['created']=f.read(OUT/'protocol.json')['created'];assert protocol==f.read(OUT/'protocol.json')
    else:f.save(OUT/'protocol.json',protocol)
    for panel in PANELS:
        for s in f.SPLITS:load(panel,s,'validation')
    f.save(OUT/'preflight.json',dict(status='passed',test_values_loaded=False,phase_audit=f.read(f.OUT/'phase_audit.json')))
    f.log('ESTIMATION PROTOCOL FROZEN',f.sha(OUT/'protocol.json'))
def check():
    for p,digest in f.read(OUT/'protocol.json')['source_hashes'].items():assert f.sha(p)==digest,p
def runfit():
    check()
    for panel in PANELS:
        for s in f.SPLITS:
            folder=OUT/'fits'/f'{panel}_f{s["fold"]}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'selection.json').exists():continue
            q=load(panel,s,'validation');rows=[]
            for person in s['validation']:
                tr=q[~q.Participant.eq(person)];va=q[q.Participant.eq(person)];models=fit(tr)
                for name,model in models.items():rows.append(dict(Participant=person,method=name,MAE=float(f.score_person(va,f.predict(va,0,model)).iloc[0])))
            cv=pd.DataFrame(rows);cv.to_csv(folder/'cv_score.csv',index=False)
            wide=cv.pivot(index='Participant',columns='method',values='MAE').reindex(columns=CANDIDATES)
            means=wide.mean();ses=wide.std(ddof=1)/np.sqrt(8);best=means.idxmin();cut=float(means[best]+ses[best]);selected=next(n for n in CANDIDATES if means[n]<=cut+1e-12)
            models=fit(q);f.save(folder/'models.json',models)
            f.save(folder/'selection.json',dict(panel=panel,fold=s['fold'],protocol_sha=f.sha(OUT/'protocol.json'),cv_mean=means.to_dict(),cv_se=ses.to_dict(),
                best_cv=best,threshold=cut,selected=selected,models_sha=f.sha(folder/'models.json'),cv_sha=f.sha(folder/'cv_score.csv'),test_values_loaded=False))
            f.log('ESTIMATION SELECTED',panel,s['fold'],selected,'minimum',best)
    f.save(OUT/'selection_complete.json',dict(status='passed',groups=15,test_values_loaded=False))
def report():
    check();assert f.read(OUT/'selection_complete.json')['status']=='passed'
    frames=[];weights=[];select=[]
    for panel in PANELS:
        for s in f.SPLITS:
            folder=OUT/'fits'/f'{panel}_f{s["fold"]}';sel=f.read(folder/'selection.json');models=f.read(folder/'models.json');assert sel['models_sha']==f.sha(folder/'models.json');select.append(sel)
            q=load(panel,s,'test')
            for method in CANDIDATES+['ValSelected']+f.FAMILIES:
                a=q.drop(columns=f.FAMILIES).copy();a['method']=method
                a['prediction']=q[method] if method in f.FAMILIES else f.predict(q,0,models[sel['selected'] if method=='ValSelected' else method]);frames.append(a)
            for method,model in models.items():
                for phase in ['global',*f.PHASES]:weights.append(dict(panel=panel,fold=s['fold'],method=method,phase=phase,**dict(zip(f.FAMILIES,model['phase'].get('0',{}).get(phase,model['global'])))))
    q=pd.concat(frames,ignore_index=True);assert not q.duplicated(['panel','method','source_excel_row']).any()
    q['error']=q.prediction-q.truth;q['ae']=q.error.abs();q['se']=q.error**2
    p=q.groupby(['panel','method','Participant'],as_index=False).agg(MAE=('ae','mean'),MSE=('se','mean'),Bias=('error','mean'),origins=('ae','size'));p['RMSE']=np.sqrt(p.MSE)
    summary=p.groupby(['panel','method'],as_index=False).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),Bias=('Bias','mean'),people=('Participant','size'),origins=('origins','sum'))
    assert summary.people.eq(40).all() and summary.origins.eq(17153).all()
    ref=pd.read_csv(SOURCE/'summary.csv')
    for panel in PANELS:
        np.testing.assert_allclose(summary[(summary.panel==panel)&(summary.method=='Equal')].MAE.item(),ref[(ref.panel==panel)&(ref.method=='Equal4')].MAE.item(),atol=1e-12,rtol=0)
    cis=[]
    for panel,g in p.groupby('panel'):
        w=g.pivot(index='Participant',columns='method',values='MAE')
        for name in CANDIDATES+['ValSelected']:
            delta=w[name]-w.Equal;lo,hi=f.bootstrap(delta);cis.append(dict(panel=panel,method=name,delta_vs_Equal=delta.mean(),low=lo,high=hi,wins=int((delta<-1e-12).sum()),people=len(delta)))
    pp=q.groupby(['panel','method','phase_5','Participant'],as_index=False).agg(MAE=('ae','mean'),Bias=('error','mean'),origins=('ae','size'))
    ss=pp.groupby(['panel','method','phase_5'],as_index=False).agg(MAE=('MAE','mean'),Bias=('Bias','mean'),people=('Participant','size'),origins=('origins','sum'))
    pc=[]
    for (panel,phase),g in pp.groupby(['panel','phase_5']):
        w=g.pivot(index='Participant',columns='method',values='MAE')
        for name in CANDIDATES+['ValSelected']:
            delta=w[name]-w.Equal;lo,hi=f.bootstrap(delta);pc.append(dict(panel=panel,phase_5=phase,method=name,delta_vs_Equal=delta.mean(),low=lo,high=hi,people=len(delta)))
    for name,data in [('summary',summary),('participant',p),('paired_ci',pd.DataFrame(cis)),('phase_summary',ss),('phase_participant',pp),('phase_paired_ci',pd.DataFrame(pc)),('weights',pd.DataFrame(weights))]:data.to_csv(OUT/(name+'.csv'),index=False)
    q.to_csv(OUT/'predictions_all.csv.gz',index=False);f.save(OUT/'selections.json',select)
    f.save(OUT/'verification.json',dict(status='passed',protocol_sha=f.sha(OUT/'protocol.json'),equal_replay=True,all17153origins=True,
        choices_frozen_before_test_loading=True,no_base_training=True,not_independent_nested_OOF=True,
        output_hashes={str(OUT/name):f.sha(OUT/name) for name in ['summary.csv','predictions_all.csv.gz','selections.json','weights.csv']}))
    print(summary.to_string(index=False));f.log('ESTIMATION COMPLETE')
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);cpus=psutil.Process().cpu_affinity();psutil.Process().cpu_affinity(cpus[8:10] if len(cpus)>=10 else cpus[-2:]);threadpool_limits(2)
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['prepare','all']:prepare()
    if mode in ['fit','all']:runfit()
    if mode in ['report','all']:report()
