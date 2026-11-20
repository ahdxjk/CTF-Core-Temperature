"""Two frozen CTF context extensions, without changing archived experiments."""
from pathlib import Path
import sys, os, json, time, argparse, datetime, msvcrt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PARENT = ROOT / 'outputs/ctf_direct_forecast_20261113'
H10 = ROOT / 'outputs/ctf_direct_forecast_h10_20261114'
sys.path.insert(0, str(PARENT))
import run_forecast as base
sys.path.insert(0, str(OUT))
import activity_plan_features as af
import numpy as np
import pandas as pd
import torch
from torch import nn

HORIZONS = [5, 10, 15, 30]
CONDITIONS = ['History', 'SimplePlan']
FAMILIES = ['MSCA', 'HE', 'PFN', 'ET']
COUNTS = {5: 16389, 10: 15619, 15: 14845, 30: 12534}
CONTEXT_DIM = None
CURRENT_CONTEXT_PATH = None
ORIGINAL_BUILD = base.build
ORIGINAL_COMMIT = base.commit


def build(kind):
    # Instantiate the archived encoder exactly as before; only input width of
    # the existing linear readout changes. No new hidden layer or fine tuning.
    net = ORIGINAL_BUILD(kind)
    width = 88 if kind == 'MSCA' else 64
    net.readout = nn.Linear(width + CONTEXT_DIM, 1)
    return net


def commit(tag, model, frames, extra):
    extra = dict(extra, condition=base.OUT.name, context_file=str(CURRENT_CONTEXT_PATH),
                 context_sha=base.sha(CURRENT_CONTEXT_PATH), context_width=CONTEXT_DIM)
    return ORIGINAL_COMMIT(tag, model, frames, extra)


base.build = build
base.commit = commit
base.HORIZONS = HORIZONS
base.KINDS = ['MSCA', 'HE']


def dataset():
    d = base.load()
    assert len(d) == 17264 and d.Corerectal.notna().sum() == 17153
    assert d.Participant.nunique() == 40 and d.trial.nunique() == 155
    table = pd.concat([base.f.table_features(g) for _, g in d.groupby('trial', sort=False)]).sort_index()
    table = table.replace([np.inf, -np.inf], np.nan)
    assert table.shape == (17264, 324)
    yy, targets, _, vv = base.f.labels(d)
    ys = {h: yy[:, base.f.HORIZONS.index(h)] for h in HORIZONS}
    dst = {h: targets[:, base.f.HORIZONS.index(h)] for h in HORIZONS}
    valid = {h: vv[:, base.f.HORIZONS.index(h)] for h in HORIZONS}
    assert {h: int(valid[h].sum()) for h in HORIZONS} == COUNTS
    context = af.build_context(d)
    features = {(condition, h): af.features_from_context(context, h, condition)
                for condition in CONDITIONS for h in HORIZONS}
    for (condition, h), x in features.items():
        assert x.index.equals(d.index) and np.isfinite(x.to_numpy()).all()
        assert len(x.columns) == (6 if condition == 'History' else 11)
        if condition == 'SimplePlan':
            np.testing.assert_array_equal(x.iloc[:, :6], features['History', h])
    return d, table, ys, dst, valid, context, features


def context_scaling(d, s, h, condition, features, valid):
    x = features[condition, h]
    idx = np.flatnonzero(d.Participant.isin(s['train']) & valid[h])
    values = x.to_numpy(dtype=np.float64)
    mean = values[idx].mean(axis=0)
    std = values[idx].std(axis=0, ddof=0)
    std[std < 1e-8] = 1.
    scaled = ((values - mean) / std).astype(np.float32)
    schema = dict(condition=condition, horizon=h, fold=s['fold'], columns=list(x.columns),
                  mean=mean.tolist(), std=std.tolist(), fit_people=s['train'], fit_rows=len(idx),
                  fit_source_rows_sha=__import__('hashlib').sha256(d.source_excel_row.iloc[idx].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
                  rules='Added context only: train-role eligible-origin mean/population SD; constant SD=1; unknown values=0 with explicit flags before scaling. Current activity flags retain original 0/1.',
                  neural_order=['current_work', 'current_activity_known'] + list(x.columns),
                  tabular_order='Archived 324 physiological/current-activity features followed by same standardized added context.')
    return pd.DataFrame(scaled, index=x.index, columns=x.columns), schema


def activate(condition, s, h, features, valid, d):
    global CONTEXT_DIM, CURRENT_CONTEXT_PATH
    base.OUT = OUT / condition
    x, schema = context_scaling(d, s, h, condition, features, valid)
    CONTEXT_DIM = 2 + x.shape[1]
    CURRENT_CONTEXT_PATH = base.OUT / 'contexts' / f'h{h}_f{s["fold"]}.json'
    assert base.read(CURRENT_CONTEXT_PATH) == schema
    return x


def add_neural_context(packed, x):
    p = dict(packed)
    safe = p['rows'].clip(min=0)
    p['activity'] = np.concatenate([packed['activity'], x.to_numpy()[safe]], axis=-1).astype(np.float32)
    p['activity'][p['rows'] < 0] = 0
    return p


def baseline_frames(h, role='test'):
    folder = H10 if h == 10 else PARENT
    for s in base.SPLITS:
        members = []
        for kind in FAMILIES:
            tags = [f'{kind}_h{h}_s{seed}_f{s["fold"]}' for seed in base.SEEDS] if kind in base.KINDS else [f'{kind}_h{h}_f{s["fold"]}']
            qs = [pd.read_csv(folder / 'predictions' / f'{tag}_{role}.csv.gz').sort_values('source_excel_row').reset_index(drop=True) for tag in tags]
            q = qs[0].copy()
            for other in qs[1:]:
                for c in ['source_excel_row', 'target_excel_row', 'Participant']:
                    np.testing.assert_array_equal(q[c], other[c])
                np.testing.assert_allclose(q.truth, other.truth, atol=1e-10, rtol=0)
            q['prediction'] = np.mean([z.prediction.to_numpy() for z in qs], axis=0)
            members.append(q)
        q = members[0].copy()
        for other in members[1:]:
            np.testing.assert_array_equal(q.source_excel_row, other.source_excel_row)
            np.testing.assert_allclose(q.truth, other.truth, atol=1e-10, rtol=0)
        q['prediction'] = np.mean([z.prediction.to_numpy() for z in members], axis=0)
        q['condition'] = 'CurrentOnly'; q['method'] = 'CTF'; q['horizon'] = h
        yield q


def dependencies():
    parent = base.read(PARENT / 'protocol.json')
    names = set()
    for group in ('data_hashes', 'code_hashes'):
        for name, expected in parent[group].items():
            assert base.sha(name) == expected, ('Archived dependency changed', name)
            names.add(name)
    names.update(str(p) for p in [Path(__file__), OUT/'activity_plan_features.py',
        PARENT/'protocol.json', H10/'protocol.json', ROOT/'revision_schedule_v3.py',
        ROOT/'outputs/strict_confirmation_v1/protocol.json', ROOT/'strict_confirmation_data_v1.py',
        ROOT/'prepare_multikernel_backbone_v1.py', ROOT/'tabpfn_memory_runtime_v1.py', OUT/'context_ledger.csv.gz'])
    for folder in (PARENT, H10):
        for name in ('verification.json', 'summary.csv'):
            names.add(str(folder/name))
        # Preserve and verify all CTF member predictions used as the old row.
        for p in (folder/'predictions').glob('*_test.csv.gz'):
            if p.name.split('_')[0] in FAMILIES:
                names.add(str(p))
    names.update(str(p) for p in (OUT/'input_features').glob('*.csv.gz'))
    for condition in CONDITIONS:
        names.update(str(p) for p in (OUT/condition/'contexts').glob('*.json'))
    return {name: base.sha(name) for name in sorted(names)}


def check():
    proto = base.read(OUT/'protocol.json')
    for name, expected in proto['dependency_hashes'].items():
        assert base.sha(name) == expected, ('Frozen dependency changed', name)
    assert base.read(OUT/'preflight.json')['status'] == 'passed'
    for condition in CONDITIONS:
        assert base.read(OUT/condition/'protocol.json') == dict(proto, active_condition=condition)
    return proto


def prepare(stuff):
    d, table, ys, dst, valid, context, features = stuff
    for condition in CONDITIONS:
        for sub in ['records', 'predictions', 'models', 'checkpoints', 'history', 'contexts']:
            (OUT/condition/sub).mkdir(parents=True, exist_ok=True)
    (OUT/'input_features').mkdir(exist_ok=True)
    for (condition, h), x in features.items():
        path = OUT/'input_features'/f'{condition}_h{h}.csv.gz'
        # Stable gzip timestamp for repeatable freeze verification.
        x.assign(source_excel_row=d.source_excel_row).to_csv(path, index=False, compression={'method':'gzip','mtime':0})
        for s in base.SPLITS:
            _, schema = context_scaling(d, s, h, condition, features, valid)
            path = OUT/condition/'contexts'/f'h{h}_f{s["fold"]}.json'
            if path.exists(): assert base.read(path) == schema
            else: base.save(path, schema)
    context.assign(source_excel_row=d.source_excel_row).to_csv(OUT/'context_ledger.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    for h in HORIZONS:
        for q, s in zip(baseline_frames(h), base.SPLITS):
            ix = np.flatnonzero(valid[h] & d.Participant.isin(s['test']))
            np.testing.assert_array_equal(q.source_excel_row, d.source_excel_row.iloc[ix])
            np.testing.assert_array_equal(q.target_excel_row, d.source_excel_row.iloc[dst[h][ix]])
            np.testing.assert_allclose(q.truth, ys[h][ix], atol=1e-10, rtol=0)
    # Feature audits are independent of targets and future realized activities.
    feature_audit = af.audit_features(d)
    assert feature_audit['status'] == 'passed'
    base.save(OUT/'feature_audit.json', feature_audit)
    packed = base.pack(d, base.SPLITS[0]); checks = []
    for condition in CONDITIONS:
        for h in (5, 30):
            x = activate(condition, base.SPLITS[0], h, features, valid, d)
            p = add_neural_context(packed, x)
            for kind in base.KINDS:
                torch.manual_seed(base.SEEDS[0]); net = build(kind).cuda().eval()
                xx = torch.tensor(p[kind][:2], device='cuda')
                aa = torch.tensor(p['activity'][:2], device='cuda'); cut=25
                with torch.no_grad():
                    a = net(xx, aa)
                    changed = xx.clone(); changed[:,cut:]=999
                    ctx = aa.clone(); ctx[:,cut:]=999
                    future_error=float((net(changed,ctx)[:,:cut]-a[:,:cut]).abs().max())
                    prefix_error=float((net(xx[:,:cut],aa[:,:cut])-a[:,:cut]).abs().max())
                assert max(future_error,prefix_error)<5e-5
                net.train(); net(xx,aa).mean().backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in net.parameters())
                checks.append(dict(condition=condition,horizon=h,kind=kind,context_width=CONTEXT_DIM,
                    parameters=sum(v.numel() for v in net.parameters()),future_error=future_error,prefix_error=prefix_error))
                del net,xx,aa; base.release()
    parent=base.read(PARENT/'protocol.json')
    proto=dict(version='CTF-observed-activity-and-simple-plan-20261114-v1',created='2026-11-14',
        dependency_hashes=dependencies(),splits=base.SPLITS,seeds=base.SEEDS,horizons=HORIZONS,
        conditions=CONDITIONS,panel='Neck14',channels=base.f.CHANNELS,origins={str(h):COUNTS[h] for h in HORIZONS},
        feature_definition=af.protocol_description(),
        history_columns=list(features['History',5].columns),plan_columns=list(features['SimplePlan',5].columns[6:]),
        neural='Unchanged MSCA/HE causal physiological encoders; expand existing linear readout to concatenate current activity and standardized current-origin context. No additional hidden layers or context recurrence.',
        neural_training=parent['neural_training'],tabular=parent['tabular']+' Additional context is identical to the neural current-origin context.',
        fusion=parent['fusion'],statistics=parent['statistics'],
        scope='Two newly fitted conditions, four horizons; old CurrentOnly four-family results reused only after exact source/target/participant checks. No mask-budget experiments or task-intensity features.',
        expected_records=240,fits=dict(neural=160,PFN=40,ET=40),
        evaluation='Same-origin participant-held-out comparisons. No future measured physiology, future realized activity, true core history, phase, absolute experiment clock, trial/condition/person ID as model predictors. Protocol metadata used only to recover legitimately announced single-block plans.',
        plan_scope='Dataset provides documented intentions rather than a field log of worker announcements; availability strata and changes in planned inputs are reported, not universal worker deployment claims.',
        python=sys.executable,torch_version=torch.__version__)
    if (OUT/'protocol.json').exists(): assert base.read(OUT/'protocol.json')==proto
    else: base.save(OUT/'protocol.json',proto)
    for condition in CONDITIONS: base.save(OUT/condition/'protocol.json',dict(proto,active_condition=condition))
    base.save(OUT/'preflight.json',dict(status='passed',neural_checks=checks,feature_audit=feature_audit,
        matched_archived_origins=True,expected_records=240,completed=datetime.datetime.now().astimezone().isoformat()))
    base.log('PREPARED',COUNTS,'240 new fits')


def record_counts():
    result={}
    for condition in CONDITIONS:
        files=list((OUT/condition/'records').glob('*.json'))
        result[condition]={kind:sum(p.name.startswith(kind+'_') for p in files) for kind in FAMILIES}
    return result


def write_progress(mode):
    base.save(OUT/f'progress_{mode}.json',dict(time=datetime.datetime.now().astimezone().isoformat(),
              counts=record_counts(),expected=240))


def bootstrap(delta):
    z=np.asarray(delta,dtype=float);rng=np.random.default_rng(20261113)
    lo,hi=np.quantile(z[rng.integers(len(z),size=(10000,len(z)))].mean(1),[.025,.975])
    return dict(delta_MAE=float(z.mean()),low=float(lo),high=float(hi),people=len(z))


def report():
    proto=check(); parts=[]; records=[]
    d,table,ys,dst,valid,context,features=dataset()
    for condition in CONDITIONS:
        base.OUT=OUT/condition
        for h in HORIZONS:
            for s in base.SPLITS:
                members={}
                for kind in FAMILIES:
                    tags=[f'{kind}_h{h}_s{seed}_f{s["fold"]}' for seed in base.SEEDS] if kind in base.KINDS else [f'{kind}_h{h}_f{s["fold"]}']
                    qs=[]
                    for tag in tags:
                        assert base.done(tag),tag
                        rec=base.read(base.record(tag));records.append(rec)
                        assert base.sha(rec['context_file'])==rec['context_sha']
                        qs.append(pd.read_csv(base.OUT/'predictions'/f'{tag}_test.csv.gz').sort_values('source_excel_row').reset_index(drop=True))
                    q=qs[0].copy()
                    for z in qs[1:]:
                        np.testing.assert_array_equal(q.source_excel_row,z.source_excel_row)
                        np.testing.assert_allclose(q.truth,z.truth,atol=1e-10,rtol=0)
                    q['prediction']=np.mean([z.prediction.to_numpy() for z in qs],axis=0)
                    q['condition']=condition;q['method']=kind;members[kind]=q;parts.append(q)
                q=members['MSCA'].copy()
                for z in members.values():
                    for c in ['source_excel_row','target_excel_row','Participant']:
                        np.testing.assert_array_equal(q[c],z[c])
                    np.testing.assert_allclose(q.truth,z.truth,atol=1e-10,rtol=0)
                q['prediction']=np.mean([members[k].prediction.to_numpy() for k in FAMILIES],axis=0)
                q['method']='CTF';parts.append(q)
    assert len(records)==240
    for h in HORIZONS: parts.extend(baseline_frames(h))
    q=pd.concat(parts,ignore_index=True)
    assert not q.duplicated(['condition','method','horizon','source_excel_row']).any()
    q['error']=q.prediction-q.truth;q['ae']=q.error.abs();q['se']=q.error**2
    keys=['condition','method','horizon']
    p=q.groupby(keys+['Participant'],as_index=False).agg(MAE=('ae','mean'),MSE=('se','mean'),Bias=('error','mean'),origins=('ae','size'))
    p['RMSE']=np.sqrt(p.MSE)
    summary=p.groupby(keys,as_index=False).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),Bias=('Bias','mean'),people=('Participant','size'),origins=('origins','sum'))
    assert summary.people.eq(40).all()
    for _,r in summary.iterrows(): assert r.origins==COUNTS[r.horizon]
    raw=d.set_index('source_excel_row')
    src=raw.loc[q.source_excel_row];target=raw.loc[q.target_excel_row]
    np.testing.assert_array_equal(src.Participant,target.Participant)
    np.testing.assert_array_equal(src.trial,target.trial)
    np.testing.assert_array_equal(target.minute.to_numpy()-src.minute.to_numpy(),q.horizon)
    np.testing.assert_allclose(target.Corerectal,q.truth,atol=1e-10,rtol=0)
    ctf=q[q.method.eq('CTF')]
    for h,g in ctf.groupby('horizon'):
        sets=[set(z.source_excel_row) for _,z in g.groupby('condition')]
        assert sets[0]==sets[1]==sets[2]
    comparisons=[]
    for h,g in p[p.method.eq('CTF')].groupby('horizon'):
        wide=g.pivot(index='Participant',columns='condition',values='MAE')
        for a,b in [('History','CurrentOnly'),('SimplePlan','History'),('SimplePlan','CurrentOnly')]:
            comparisons.append(dict(horizon=h,method=a,reference=b,**bootstrap(wide[a]-wide[b])))
    availability=[]; phase=[]
    for h,g in ctf.groupby('horizon'):
        plan=features['SimplePlan',h].iloc[:,6:]
        key=d.source_excel_row.to_numpy(); known=plan.iloc[:,0].to_numpy().astype(bool)
        groupmap=pd.Series(np.where(known,'available','unavailable'),index=key)
        g=g.copy();g['plan_availability']=g.source_excel_row.map(groupmap)
        z=g.groupby(['condition','plan_availability','Participant'],as_index=False).agg(MAE=('ae','mean'),origins=('ae','size'))
        availability.append(z.groupby(['condition','plan_availability'],as_index=False).agg(MAE=('MAE','mean'),people=('Participant','size'),origins=('origins','sum')).assign(horizon=h))
        z=g.groupby(['condition','phase_5','Participant'],as_index=False).agg(MAE=('ae','mean'),origins=('ae','size'))
        phase.append(z.groupby(['condition','phase_5'],as_index=False).agg(MAE=('MAE','mean'),people=('Participant','size'),origins=('origins','sum')).assign(horizon=h))
    q.to_csv(OUT/'predictions_all.csv.gz',index=False);p.to_csv(OUT/'participant.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_ci.csv',index=False)
    pd.concat(availability).to_csv(OUT/'plan_availability_summary.csv',index=False)
    pd.concat(phase).to_csv(OUT/'phase_summary.csv',index=False)
    main=summary[summary.method.eq('CTF')].pivot(index='condition',columns='horizon',values='MAE').reindex(['CurrentOnly','History','SimplePlan'])
    main.to_csv(OUT/'CTF_main_table.csv')
    labels={'CurrentOnly':'生理历史＋当前活动，无未来计划','History':'生理历史＋当前活动＋活动历史，无未来计划','SimplePlan':'生理历史＋当前活动＋活动历史＋下一段简单计划'}
    lines=['# CTF 活动历史与简单计划预测实验','', 'MAE（°C），40人按人等权；Neck14；同一时距三行采用相同预测起点。','',
           '| CTF输入条件 | 5分钟 | 10分钟 | 15分钟 | 30分钟 |','|---|---:|---:|---:|---:|']
    for condition,row in main.iterrows(): lines.append('| '+labels[condition]+' | '+' | '.join(f'{row[h]:.4f}' for h in HORIZONS)+' |')
    lines+=['','活动历史与计划定义见 protocol.json 和 context_ledger.csv.gz。SimplePlan只提供单段合法名义安排，不提供整套试验日程、事后活动或强度。',
            '两种新模型均重新拟合；无计划输入不等于强制返回旧模型预测。配对区间使用重复探索过的40名受试者，未校正多重比较。','',
            '## 按人配对差值（负值表示改善）','',pd.DataFrame(comparisons).to_string(index=False)]
    (OUT/'结果表.md').write_text(chr(10).join(lines),encoding='utf-8')
    (OUT/'report.html').write_text('<meta charset="utf-8"><h1>CTF: observed activity and simple next-block plans</h1>'+main.rename(index=labels).to_html(float_format=lambda x:f'{x:.4f}')+'<h2>All members</h2>'+summary.to_html(index=False)+'<h2>Paired differences</h2>'+pd.DataFrame(comparisons).to_html(index=False),encoding='utf-8')
    base.save(OUT/'verification.json',dict(status='passed',records=240,hashes_verified=True,context_hashes_verified=True,
        matched_origins=True,measured_same_trial_future_targets=True,people=40,origins=COUNTS,
        family_weights=[.25]*4,protocol_sha=base.sha(OUT/'protocol.json'),completed=datetime.datetime.now().astimezone().isoformat()))
    base.log('ALL COMPLETE',main.to_string())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','gpu','et','report']);args=parser.parse_args()
    base.runtime(args.mode)
    if args.mode=='report': report();return
    stuff=dataset()
    if args.mode=='prepare': prepare(stuff);return
    check();d,table,ys,dst,valid,context,features=stuff
    with (OUT/f'{args.mode}.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        for s in base.SPLITS:
            packed=base.pack(d,s) if args.mode=='gpu' else None
            for h in HORIZONS:
                for condition in CONDITIONS:
                    x=activate(condition,s,h,features,valid,d)
                    if args.mode=='gpu':
                        p=add_neural_context(packed,x)
                        for kind in base.KINDS:
                            for seed in base.SEEDS: base.neural(d,s,h,kind,seed,p,ys,dst,valid)
                        base.table_fit(d,pd.concat([table,x],axis=1),s,h,'PFN',ys,dst,valid)
                        del p;base.release()
                    else: base.table_fit(d,pd.concat([table,x],axis=1),s,h,'ET',ys,dst,valid)
                    write_progress(args.mode)
            if packed is not None: del packed;base.release()
    base.save(OUT/f'{args.mode}_complete.json',dict(status='complete',counts=record_counts(),completed=datetime.datetime.now().astimezone().isoformat()))


if __name__=='__main__': main()
