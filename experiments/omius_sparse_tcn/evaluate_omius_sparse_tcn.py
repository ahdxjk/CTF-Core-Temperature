"""Score frozen Sparse5 TCNs and replay the standard GRU/TCN on identical endpoints."""
import numpy as np
import pandas as pd
import torch
from run_omius_sparse_tcn import ROOT, OUT, SETTING, TCN, flatten_inputs, r


def main():
    r.COLS = ['head', 'HR']
    torch.set_num_threads(2)
    protocol = r.sha(OUT/'protocol.json')
    assert r.read(OUT/'training_complete.json') == dict(fits=10, protocol_sha=protocol)
    peer = ROOT/'outputs/external_peer_completion_v3'
    assert r.sha(peer/'data.pkl') == r.read(peer/'preflight.json')['data_sha']
    data = pd.read_pickle(peer/'data.pkl').query("dataset == 'Omius'")
    keys = ['Participant','trial','minute','source_excel_row','Condition','Corerectal']
    table = pd.read_csv(peer/'predictions/HeadHR2_fold0_ExtraTrees_external.csv.gz')[keys].copy()
    eligible = data.Corerectal.notna() & data.HR.notna() & data['head'].notna().groupby(data.trial).cummax()
    observed = data.loc[eligible, keys].set_index('source_excel_row')
    ref = table.set_index('source_excel_row')
    assert observed.index.is_unique and len(observed) == len(ref) == 160
    pd.testing.assert_frame_equal(observed.sort_index(),ref.sort_index(),check_dtype=False)
    assert table.Participant.nunique() == 10 and table.trial.nunique() == 20
    assert table.Condition.value_counts().to_dict() == {'OH':80,'SH':80}
    fits = []
    for fold in range(5):
        for seed in r.SEEDS:
            tag = r.fit_tag(SETTING, fold, seed)
            rec = r.checked_record(OUT/'records'/(tag+'.json'))
            ck = torch.load(rec['models']['Robust']['path'], map_location='cpu',weights_only=False)
            assert ck['protocol_sha'] == protocol and ck['fold']==fold and ck['seed']==seed
            assert ck['policy']=='Robust' and ck['setting']==SETTING and ck['channels']==r.COLS
            m = TCN(8,64).eval()
            m.load_state_dict(ck['state'])
            series = []
            with torch.inference_mode():
                for _,g in data.groupby('trial',sort=False):
                    g = g.sort_values('minute')
                    raw = r.encode_trial(g,ck['scale'])
                    x = torch.from_numpy(flatten_inputs(raw[None],'Raw4'))
                    p = m(x)[0].numpy()*ck['scale']['ystd'] + ck['scale']['ymean']
                    series.append(pd.Series(p,index=g.source_excel_row))
            table[tag] = pd.concat(series).loc[ref.index].to_numpy()
            fits.append(tag)
    table[SETTING] = table[fits].to_numpy(dtype=float).mean(axis=1)
    for baseline in ['GRUFull','TCN']:
        folds = []
        for fold in range(5):
            name = f'HeadHR2_fold{fold}_{baseline}'
            path = peer/'predictions'/(name+'_external.csv.gz')
            assert r.sha(path)==r.read(peer/'records'/(name+'.json'))['hashes'][str(path)]
            df = pd.read_csv(path).set_index('source_excel_row').loc[ref.index]
            pd.testing.assert_frame_equal(df[ref.columns],ref)
            folds.append(df.prediction.to_numpy())
        table[baseline] = np.mean(folds,axis=0)
    metrics = []
    for condition,g in [('both',table),*table.groupby('Condition')]:
        for method in [SETTING,'GRUFull','TCN']:
            err = g[method]-g.Corerectal
            assert np.isfinite(err).all()
            person = g.assign(ae=err.abs(),se=err**2,bias=err).groupby('Participant')
            scores = person[['ae','se','bias']].mean()
            scores['rmse'] = np.sqrt(scores.se)
            scores['n'] = person.size()
            metrics.append(scores.assign(condition=condition,method=method).reset_index())
    per = pd.concat(metrics,ignore_index=True)
    summary = per.groupby(['condition','method']).agg(MAE=('ae','mean'), RMSE=('rmse','mean'),
                        bias=('bias','mean'),people=('Participant','size'),points=('n','sum'))
    archived = pd.read_csv(peer/'summary.csv').query("dataset == 'Omius'").set_index('method')
    for method in ['GRUFull','TCN']:
        np.testing.assert_allclose(summary.loc[('both',method),['MAE','RMSE']],
                                   archived.loc[method,['MAE','RMSE']].astype(float),atol=1e-9,rtol=0)
    table.to_csv(OUT/'predictions_omius.csv',index=False)
    per.to_csv(OUT/'participant_metrics.csv',index=False)
    summary.to_csv(OUT/'summary.csv')
    print(summary.to_string())


if __name__ == '__main__':
    main()
