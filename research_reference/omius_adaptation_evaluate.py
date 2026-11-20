"""Evaluate frozen HeadHR2 cadence adaptations on the archived 160 Omius endpoints.

No external observations or labels participate in training, model selection,
scaling, or ensemble weighting. All completed source fits are required first.
"""
from pathlib import Path
import hashlib,json,os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import numpy as np
import pandas as pd
import torch
import run_regional_neural_transfer_v2 as train
from channel_bank400_v1 import ExpandedTCN,trial_features
from regional_neural_transfer_v2_model import HeldAgeMultikernelBackbone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/omius_head2_adapt_20261117'
V3=ROOT/'outputs/external_peer_completion_v3'
SETTINGS=['MSCA_HeldAgeRateMix','HE_RateMix','HE_Sparse5']
SEEDS=[20261103,20261104]
COLS=['head','HR']
DEVICE='cpu' if '--cpu' in sys.argv else 'cuda'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
train.COLS=COLS

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    # The full source training set must be complete before touching external data.
    marker=OUT/'training_complete.json'
    assert marker.exists(),'All 30 source-only models must be completed before external evaluation'
    finished=json.loads(marker.read_text(encoding='utf8'));assert finished['status']=='complete'
    artifacts={}
    for f in range(5):
        for setting in SETTINGS:
            for seed in SEEDS:
                tag=f'{setting}_f{f}_s{seed}'
                rec=OUT/'records'/f'{tag}.json';ck=OUT/'models'/f'{tag}_Robust.pt'
                assert rec.exists() and ck.exists(),tag
                r=json.loads(rec.read_text(encoding='utf8'));assert r['status']=='passed'
                expected=r['models']['Robust']['sha256'];assert sha(ck)==expected
                artifacts[tag]=expected
    source_preflight=json.loads((V3/'preflight.json').read_text(encoding='utf8'))
    assert source_preflight['status']=='passed' and source_preflight['data_sha']==sha(V3/'data.pkl')
    data=pd.read_pickle(V3/'data.pkl');data=data[data.dataset.eq('Omius')].copy()
    assert len(data)==1420 and data.trial.nunique()==20 and data.Participant.nunique()==10
    reference=pd.read_csv(V3/'predictions/HeadHR2_fold0_ExtraTrees_external.csv.gz')
    keys=['Participant','trial','minute','Corerectal','source_excel_row','dataset','Condition']
    assert len(reference)==160 and reference.source_excel_row.is_unique
    assert reference.Condition.value_counts().to_dict()=={'OH':80,'SH':80}
    rows=data.set_index('source_excel_row').loc[reference.source_excel_row]
    for c in ['Participant','trial','minute','Condition']:
        np.testing.assert_array_equal(rows[c].to_numpy(),reference[c].to_numpy())
    np.testing.assert_allclose(rows.Corerectal,reference.Corerectal,rtol=0,atol=1e-12)
    ever=data['head'].notna().groupby(data.trial).cummax()
    mask=data.Corerectal.notna() & data.HR.notna() & ever
    assert set(data.loc[mask,'source_excel_row'])==set(reference.source_excel_row)
    groups=[g.sort_values('minute') for _,g in data.groupby('trial',sort=False)]
    predictions={};metadata={}
    for fold in range(5):
        for model in ['TabPFN','ExtraTrees','MSCA','HE_TCN','BlendFixed']:
            path=V3/'predictions'/f'HeadHR2_fold{fold}_{model}_external.csv.gz'
            r=json.loads((V3/'records'/f'HeadHR2_fold{fold}_{model}.json').read_text(encoding='utf8'))
            assert r['status']=='passed' and r['hashes'][str(path)]==sha(path)
            q=pd.read_csv(path);pd.testing.assert_frame_equal(q[keys],reference[keys])
            predictions[(fold,model)]=q.prediction.to_numpy();metadata[str(path)]=sha(path)
        native=np.mean([predictions[(fold,k)] for k in ['TabPFN','ExtraTrees','MSCA','HE_TCN']],axis=0)
        np.testing.assert_allclose(native,predictions[(fold,'BlendFixed')],atol=5e-6,rtol=0)
        for setting in SETTINGS:
            seed_predictions=[]
            for seed in SEEDS:
                tag=f'{setting}_f{fold}_s{seed}';path=OUT/'models'/f'{tag}_Robust.pt'
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['fold']==fold and ck['seed']==seed and ck['setting']==setting and ck['policy']=='Robust' and ck['channels']==COLS
                scale=ck['scale'];model=(HeldAgeMultikernelBackbone() if setting.startswith('MSCA') else ExpandedTCN(2,'Direct400')).to(DEVICE)
                model.load_state_dict(ck['state']);model.eval();outputs=[]
                with torch.inference_mode():
                    for g in groups:
                        raw=train.encode_trial(g,scale)
                        if setting.startswith('MSCA'):a=train.msca_input(raw[None],'HeldAge')
                        else:a=trial_features(raw)[None]
                        pred=model(torch.from_numpy(a).to(DEVICE)).cpu().numpy()[0]*scale['ystd']+scale['ymean']
                        outputs.append(pd.Series(pred,index=g.source_excel_row))
                out=pd.concat(outputs);assert out.index.is_unique
                v=out.loc[reference.source_excel_row].to_numpy();assert np.isfinite(v).all()
                seed_predictions.append(v)
                pd.DataFrame({'source_excel_row':reference.source_excel_row,'prediction':v}).to_csv(OUT/'predictions'/f'{tag}_Omius.csv.gz',index=False)
                del model
            predictions[(fold,setting)]=np.mean(seed_predictions,axis=0)
        print('EVALUATED FOLD',fold,flush=True)
    folds=[]
    for f in range(5):
        vals=[predictions[(f,k)] for k in ['MSCA_HeldAgeRateMix','HE_RateMix','TabPFN','ExtraTrees']]
        folds.append(np.mean(vals,axis=0))
    table=reference[keys].copy()
    table['CTF_HeldMix']=np.mean(folds,axis=0)
    table['HE_Sparse5']=np.mean([predictions[(f,'HE_Sparse5')] for f in range(5)],axis=0)
    table['CTF_Native']=np.mean([predictions[(f,'BlendFixed')] for f in range(5)],axis=0)
    table['HE_Native']=np.mean([predictions[(f,'HE_TCN')] for f in range(5)],axis=0)
    for name in ['MSCA_HeldAgeRateMix','HE_RateMix','TabPFN','ExtraTrees']:
        table[name]=np.mean([predictions[(f,name)] for f in range(5)],axis=0)
    table.to_csv(OUT/'omius_adapt_predictions.csv.gz',index=False)
    summaries=[];individual=[]
    for condition,g in [('both',table),('OH',table[table.Condition.eq('OH')]),('SH',table[table.Condition.eq('SH')])]:
        for name in ['CTF_HeldMix','HE_Sparse5','CTF_Native','HE_Native','MSCA_HeldAgeRateMix','HE_RateMix','TabPFN','ExtraTrees']:
            err=g[name]-g.Corerectal
            person=g.assign(ae=abs(err),se=err**2,e=err).groupby('Participant')[['ae','se','e']].mean()
            summaries.append(dict(condition=condition,model=name,MAE=person.ae.mean(),RMSE=np.sqrt(person.se).mean(),bias=person.e.mean(),people=len(person),points=len(g)))
            for pid,row in person.iterrows():individual.append(dict(condition=condition,model=name,Participant=pid,MAE=row.ae,RMSE=np.sqrt(row.se),bias=row.e))
    pd.DataFrame(summaries).to_csv(OUT/'omius_adapt_summary.csv',index=False)
    individuals=pd.DataFrame(individual)
    individuals.to_csv(OUT/'omius_adapt_person_metrics.csv',index=False)
    comparisons=[]
    for condition,g in individuals.groupby('condition'):
        wide=g.pivot(index='Participant',columns='model',values='MAE')
        for proposed,baseline in [('CTF_HeldMix','CTF_Native'),('HE_Sparse5','HE_Native')]:
            delta=(wide[proposed]-wide[baseline]).to_numpy();rng=np.random.default_rng(20261117)
            boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            comparisons.append(dict(condition=condition,model=proposed,reference=baseline,delta_mae=delta.mean(),ci_low=lo,ci_high=hi,people=len(delta),improved=int((delta<0).sum())))
    pd.DataFrame(comparisons).to_csv(OUT/'omius_adapt_paired.csv',index=False)
    report=dict(status='complete',source_only=True,source_folds=5,seeds=SEEDS,setting_selection='predeclared, robust source validation',
       formula='per fold mean(MSCA2seed,HE2seed,PFN,ExtraTrees); then mean 5 folds',
       support='160 paired Omius endpoints, 80 active +80 sham; 20 full causal trial histories',
       metrics='point errors averaged within participant then across 10 participants',model_sha256=artifacts,
       tabular_prediction_sha256=metadata,results_sha256=sha(OUT/'omius_adapt_predictions.csv.gz'),script_sha256=sha(__file__))
    (OUT/'omius_evaluation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__': main()
