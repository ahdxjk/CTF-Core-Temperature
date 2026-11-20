"""Re-score saved, person-held-out predictions using fixed activity-only phases.

No training, sensor selection, or phase-boundary selection is performed here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/five_phase_v1/scores'
LABEL = ROOT / 'outputs/five_phase_v1/labels/minute_labels.csv'
NAMES = {'P1':'初始静息', 'P2':'运动开始早期', 'P3':'持续运动',
         'P4':'早期恢复', 'P5':'后续恢复', 'U':'无法判定', 'ALL':'整体'}


def load_predictions():
    cols = ['source_excel_row','Participant','Condition','trial','minute','Corerectal','prediction']
    parts=[]
    src=ROOT/'outputs/guarded_correction_v1/all_test_predictions.csv'
    q=pd.read_csv(src)
    for old,new in [('CatBaseline','CatBoost_12'),('MemoryEnsemble','Memory_12'),('PredictionGate','Gate_12')]:
        a=q.loc[q.model.eq(old),cols].copy()
        a['model']=new; a['protocol']='SeenCondition'; parts.append(a)
    q=pd.read_csv(ROOT/'outputs/phase_distill_v2/all_test_predictions.csv')
    for old,new in [('CatBoost_2signals_history','CatBoost_2'),('Distilled_2signals_history','Distilled_2')]:
        a=q.loc[q.model.eq(old),cols].copy()
        a['model']=new; a['protocol']='SeenCondition'; parts.append(a)
    q=pd.read_csv(ROOT/'outputs/leave_condition_v1/all_ensemble_predictions.csv')
    mapping={'Cat':'CatBoost_12','Gate':'Gate_12','Memory':'Memory_12',
             'CompactAccuracy':'JointMLP_12','HistoryHuberScene':'HistoryHuber_12'}
    for (protocol,model),g in q.groupby(['protocol','model']):
        if model not in mapping or (protocol=='SeenCondition' and model in ['Cat','Gate']):
            continue
        a=g[cols].copy();a['model']=mapping[model];a['protocol']=protocol;parts.append(a)
    return pd.concat(parts,ignore_index=True)


def metrics(g):
    p=g.groupby('Participant').agg(mae=('ae','mean'),mse=('se','mean'),
        bias=('error','mean'),over03=('over03','mean'))
    return dict(people=len(p),trials=g.trial.nunique(),points=len(g),
        subject_MAE=p.mae.mean(),subject_RMSE=np.sqrt(p.mse).mean(),
        subject_bias=p.bias.mean(),subject_fraction_abs_gt03=p.over03.mean(),
        pooled_MAE=g.ae.mean(),pooled_RMSE=np.sqrt(g.se.mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lab=pd.read_csv(LABEL)
    assert len(lab)==17264 and lab.source_excel_row.is_unique
    pred=load_predictions()
    meta=pd.read_csv(ROOT/'outputs/cpu_baselines_v1/metadata.csv')
    keys=set(meta.source_excel_row)
    for (protocol,model),g in pred.groupby(['protocol','model']):
        assert len(g)==17162 and g.source_excel_row.is_unique
        assert set(g.source_excel_row)==keys,(protocol,model)
        chk=g.merge(meta,on='source_excel_row',validate='one_to_one',suffixes=('_p','_m'))
        for c in ['Participant','Condition','trial','minute','Corerectal']:
            assert np.allclose(chk[c+'_p'],chk[c+'_m']),(model,c)
    check=meta.merge(lab,on='source_excel_row',validate='one_to_one',suffixes=('_m','_l'))
    for c in ['Participant','Condition','trial','minute']:
        assert np.allclose(check[c+'_m'],check[c+'_l']),c
    pred=pred.merge(lab[['source_excel_row','phase_5','phase_10','phase_15']],
                    on='source_excel_row',validate='many_to_one')
    assert pred[['phase_5','phase_10','phase_15']].notna().all().all()
    pred['error']=pred.prediction-pred.Corerectal
    pred['ae']=pred.error.abs();pred['se']=pred.error**2;pred['over03']=pred.ae.gt(.3)
    rows=[];person=[];trial=[]
    for width in [5,10,15]:
        for (protocol,model),g in pred.groupby(['protocol','model']):
            for ph,h in [('ALL',g)]+list(g.groupby('phase_'+str(width))):
                rows.append(dict(window_min=width,protocol=protocol,model=model,phase=ph,
                                 phase_name=NAMES[ph],**metrics(h)))
                for pid,z in h.groupby('Participant'):
                    person.append(dict(window_min=width,protocol=protocol,model=model,
                        phase=ph,Participant=pid,points=len(z),mae=z.ae.mean(),bias=z.error.mean()))
                if width==5:
                    for tid,z in h.groupby('trial'):
                        trial.append(dict(protocol=protocol,model=model,phase=ph,trial=tid,
                            Participant=int(z.Participant.iloc[0]),Condition=int(z.Condition.iloc[0]),
                            points=len(z),mae=z.ae.mean(),bias=z.error.mean()))
    scores=pd.DataFrame(rows);persons=pd.DataFrame(person)
    scores.to_csv(OUT/'phase_scores.csv',index=False,encoding='utf-8-sig')
    persons.to_csv(OUT/'per_person_phase.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(trial).to_csv(OUT/'per_trial_phase.csv',index=False,encoding='utf-8-sig')
    pairs=[];rng=np.random.default_rng(20261101)
    # Descriptive paired-person contrasts, not independent confirmation or causation.
    for (w,protocol,model),g in persons.groupby(['window_min','protocol','model']):
        p=g.pivot(index='Participant',columns='phase',values='mae')
        for a,b in [('P2','P3'),('P4','P3'),('P5','P3'),('P4','P5')]:
            z=p[[a,b]].dropna();d=(z[a]-z[b]).to_numpy()
            if not len(d):continue
            boots=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            pairs.append(dict(window_min=w,protocol=protocol,model=model,phase=a,reference=b,
                paired_people=len(d),mean_difference=d.mean(),ci_low=np.quantile(boots,.025),
                ci_high=np.quantile(boots,.975),people_with_higher_error=int((d>0).sum())))
    pd.DataFrame(pairs).to_csv(OUT/'paired_phase_contrasts.csv',index=False,encoding='utf-8-sig')
    # Prespecified conservative audit sensitivity after checking activity/workrest
    # discrepancies, not after looking at their prediction errors. Original rows stay intact.
    sensitivities=[]
    for exclusion, tids in [('exclude_isolated_work_flag',[115]),
                            ('exclude_three_reviewed_trials',[115,133,147])]:
        for (protocol,model),g in pred[~pred.trial.isin(tids)].groupby(['protocol','model']):
            for ph,h in [('ALL',g)]+list(g.groupby('phase_5')):
                sensitivities.append(dict(exclusion=exclusion,excluded_trials=';'.join(map(str,tids)),
                    protocol=protocol,model=model,phase=ph,**metrics(h)))
    pd.DataFrame(sensitivities).to_csv(OUT/'activity_review_sensitivity.csv',index=False,encoding='utf-8-sig')
    valid=lab.merge(meta[['source_excel_row','Corerectal']],on='source_excel_row',how='left',validate='one_to_one')
    coverage=[]
    for width in [5,10,15]:
        for ph,g in valid.groupby('phase_'+str(width)):
            good=g[g.Corerectal.notna()]
            coverage.append(dict(window_min=width,phase=ph,phase_name=NAMES[ph],
                people=g.Participant.nunique(),trials=g.trial.nunique(),record_points=len(g),
                valid_label_points=len(good),valid_people=good.Participant.nunique(),
                valid_trials=good.trial.nunique()))
    pd.DataFrame(coverage).to_csv(OUT/'phase_coverage.csv',index=False,encoding='utf-8-sig')
    checks={'labels':len(lab),'valid_target_points':len(meta),'people':meta.Participant.nunique(),
        'trials':meta.trial.nunique(),'all_prediction_keys_and_targets_match':True,
        'models_protocols':pred.groupby(['protocol','model']).size().to_dict(),
        'label_file_sha256':hashlib.sha256(LABEL.read_bytes()).hexdigest(),
        'training_performed':False,'boundary_selected_by_error':False,
        'status':'exploratory reanalysis of repeatedly inspected held-out predictions'}
    checks['models_protocols']={str(k):int(v) for k,v in checks['models_protocols'].items()}
    (OUT/'audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    print(scores[(scores.window_min==5)&(scores.protocol=='SeenCondition')].pivot(index='phase',columns='model',values='subject_MAE').round(4).to_string())
    print(pd.DataFrame(coverage).query('window_min==5').to_string(index=False))


if __name__=='__main__':main()
