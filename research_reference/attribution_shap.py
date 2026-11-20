"""Exact baseline Shapley for frozen All22 four-family current-temperature estimates.

Seven groups of original physical channels; channel-derived features are rebuilt
for actual and flat training-mean histories, then spliced. Observation patterns
are held fixed. No explanation model or predictor is fitted here.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = 'true'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits
from multikernel_backbone_model_v1 import build_model
from channel_bank400_v1 import ExpandedTCN, trial_features

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/interpretability_v1'
OUT = BASE / 'shap'
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / 'temp'
TMP.mkdir(exist_ok=True)
os.environ['TMP'] = os.environ['TEMP'] = str(TMP)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
PROTOCOL = json.loads((BASE / 'protocol.json').read_text(encoding='utf8'))
GROUPS = PROTOCOL['groups']
GROUP_NAMES = list(GROUPS)
CHANNELS = [c for cols in GROUPS.values() for c in cols]
G = len(GROUPS)
COALITIONS = ((np.arange(2**G)[:, None] >> np.arange(G)) & 1).astype(bool)
CH_GROUP = np.array([next(i for i, v in enumerate(GROUPS.values()) if c in v) for c in CHANNELS])
CH_COAL = COALITIONS[:, CH_GROUP]
SEEDS = [20261103, 20261104]
MEMBERS = ['TabPFN', 'MSCA', 'HE-TCN', 'ExtraTrees']
SPLITS = json.loads((ROOT / 'outputs/cpu_baselines_v1/splits.json').read_text())


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tab_features(group):
    """Verbatim mathematical feature contract of panel tabular training."""
    fields = {}
    t = group.minute
    for c in CHANNELS:
        x = group[c].astype(float)
        mask = x.notna()
        last = x.ffill()
        recorded = t.where(mask)
        fields[c+'_last'] = last
        fields[c+'_observed'] = mask.astype(float)
        fields[c+'_age'] = t-recorded.ffill()
        for n in [5, 15, 30, 60]:
            r = x.rolling(n, min_periods=1)
            fields[f'{c}_mean{n}'] = r.mean()
            fields[f'{c}_n{n}'] = r.count()
            if n <= 15:
                count = r.count()
                st = recorded.rolling(n, min_periods=1).sum()
                stt = (recorded*t).rolling(n, min_periods=1).sum()
                stx = (t*x).rolling(n, min_periods=1).sum()
                denom = stt-st**2/count
                fields[f'{c}_slope{n}'] = ((stx-st*r.sum()/count)/denom).where((count >= 2) & (denom > 1e-10))
            else:
                fields[f'{c}_ema{n}'] = last.ewm(span=n, adjust=False, min_periods=1).mean()
        for n in (5, 15, 30):
            lag = last.shift(n)
            fields[f'{c}_lag{n}'] = lag
            fields[f'{c}_delta{n}'] = last-lag
        r = x.rolling(15, min_periods=1)
        fields[c+'_range15'] = r.max()-r.min()
        fields[c+'_fast_slow'] = last.ewm(span=5, adjust=False).mean()-last.ewm(span=30, adjust=False).mean()
    return pd.DataFrame(fields, index=group.index)


def raw_encode(prefix, scale):
    vals = prefix[CHANNELS]
    last = vals.ffill()
    observed = vals.notna()
    age = np.column_stack([(prefix.minute-prefix.minute.where(observed[c]).ffill()).fillna(0).to_numpy()/60 for c in CHANNELS])
    return np.stack([((last-np.array(scale['mean']))/np.array(scale['std'])).fillna(0).to_numpy(),
                     observed.to_numpy(float), age, last.notna().to_numpy(float)], -1).astype(np.float32)


def primary_reference(prefix, means):
    ref = prefix.copy()
    for c in CHANNELS:
        ref[c] = np.where(prefix[c].notna(), means[c], np.nan)
    return ref


def tab_coalitions(prefix, ref):
    a, b = tab_features(prefix).iloc[-1], tab_features(ref).iloc[-1]
    assert len(a) == 506
    fc = np.repeat(CH_COAL, 23, axis=1)
    return pd.DataFrame(np.where(fc, a.to_numpy()[None], b.to_numpy()[None]), columns=a.index)


def neural_coalitions(prefix, ref, scale, family):
    a, b = raw_encode(prefix, scale), raw_encode(ref, scale)
    if family == 'MSCA':
        def convert(z):
            return np.stack([np.where(z[..., 1] > 0, z[..., 0], 0), z[..., 1], np.ones_like(z[..., 1])], -1)
        a, b = convert(a), convert(b)
    else:
        a, b = trial_features(a), trial_features(b)
    return a, b


@torch.inference_mode()
def neural_predict(models, prefix, ref, scale, family, device='cpu'):
    actual, baseline = neural_coalitions(prefix, ref, scale, family)
    result = []
    # Keep dense attention and 400-feature bank allocations bounded.
    for start in range(0, 2**G, 16):
        selectors = CH_COAL[start:start+16, None, :, None]
        batch = torch.from_numpy(np.where(selectors, actual[None], baseline[None])).to(device)
        pred = torch.stack([m(batch)[:, -1] for m in models]).mean(0)
        result.append((pred*scale['ystd']+scale['ymean']).cpu().numpy())
    return np.concatenate(result).astype(float)


def load_neural(fold, family, device='cpu'):
    models, scale, paths = [], None, []
    for seed in SEEDS:
        if family == 'MSCA':
            path = Path(f'REPLACE_WITH_MODEL_DIR/multikernel_backbone_v1/MK357_s{seed}_f{fold}.pt')
        else:
            path = Path(f'REPLACE_WITH_MODEL_DIR/direct400_panels_v1/fold{fold}_All22_seed{seed}.pt')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        s = ck['preprocessing'] if family == 'MSCA' else ck['scale']
        assert ck['split'] == SPLITS[fold] and s['columns'] == CHANNELS
        if scale is not None:
            assert s == scale
        scale = s
        net = build_model('MK357') if family == 'MSCA' else ExpandedTCN(len(CHANNELS), 'Direct400')
        net.load_state_dict(ck['state'])
        models.append(net.to(device).eval())
        paths.append(str(path))
    return models, scale, paths


def archived(fold, family):
    if family in ('TabPFN', 'ExtraTrees'):
        n = 'TabPFN_3_5' if family == 'TabPFN' else family
        a = pd.read_csv(ROOT / f'outputs/peer_comparison_v1/panel_tables/All22/fold{fold}_{n}_test.csv')
        return a.set_index('source_excel_row').prediction
    arrays = []
    for seed in SEEDS:
        if family == 'MSCA':
            a = pd.read_csv(ROOT / f'outputs/multikernel_backbone_v1/fits/MK357_s{seed}_f{fold}/test.csv.gz')
            arrays.append(a.loc[a.panel.eq('All22')].set_index('row').pred)
        else:
            a = pd.read_csv(ROOT / f'outputs/direct400_panels_v1/fold{fold}_All22_seed{seed}_test.csv')
            arrays.append(a.set_index('source_excel_row').prediction)
    assert arrays[0].index.equals(arrays[1].index)
    return (arrays[0]+arrays[1])/2


def frozen_tab(fold, family):
    n = 'TabPFN_3_5' if family == 'TabPFN' else family
    record = json.loads((ROOT / f'outputs/peer_comparison_v1/panel_tables/All22/fold{fold}_{n}.json').read_text(encoding='utf8'))
    assert sha(record['model_path']) == record['model_sha']
    if family == 'TabPFN':
        from tabpfn_memory_runtime_v1 import configure_pfn_chunks
        configure_pfn_chunks(65536)
        from tabpfn.model_loading import load_fitted_tabpfn_model
        model = load_fitted_tabpfn_model(record['model_path'], device='cpu')
        model.n_preprocessing_jobs = 2
        # Saved cache may materialize lazily on first predict; no training/context changes.
    else:
        model = joblib.load(record['model_path'])
        model.named_steps['model'].n_jobs = 4
    return model, record


def prepare_tab_batch(data, selected, means):
    arrays = []
    for s in selected.itertuples(index=False):
        prefix = data.loc[data.trial.eq(s.trial) & data.minute.le(s.minute)].copy()
        assert len(prefix) == s.prefix_length
        arrays.append(tab_coalitions(prefix, primary_reference(prefix, means)))
    return pd.concat(arrays, ignore_index=True)


def exact_phi(values):
    """values is (..., 128); return (..., 7) using all coalition marginal effects."""
    phi = np.zeros((*values.shape[:-1], G))
    for i in range(G):
        for subset in range(2**G):
            if subset & (1 << i):
                continue
            n = int(subset).bit_count()
            w = math.factorial(n)*math.factorial(G-n-1)/math.factorial(G)
            phi[..., i] += w*(values[..., subset | (1 << i)]-values[..., subset])
    return phi


def run_family(fold, family, data, selection, benchmark=False):
    selected = selection.loc[selection.fold.eq(fold)].copy()
    if benchmark:
        selected = selected.sort_values(['prefix_length', 'sample_id']).tail(1)
    tag = f'fold{fold}_{family.replace("-", "")}'
    if benchmark:
        tag += '_benchmark'
    result_path = OUT / (tag+'.npz')
    if result_path.exists():
        print('REUSE', tag, flush=True)
        return
    means = data.loc[data.Participant.isin(SPLITS[fold]['train']), CHANNELS].mean().fillna(0)
    started = time.perf_counter()
    print('START', tag, 'samples', len(selected), flush=True)
    if family in ('TabPFN', 'ExtraTrees'):
        queries = prepare_tab_batch(data, selected, means)
        model, record = frozen_tab(fold, family)
        # The reused Base12 table has its original column order; align by names.
        assert set(queries.columns) == set(model.feature_names_in_)
        queries = queries.loc[:, model.feature_names_in_]
        loaded = time.perf_counter()
        print('LOADED', tag, 'seconds', round(loaded-started, 2), flush=True)
        outs = []
        for start in range(0, len(queries), 256):
            q = queries.iloc[start:start+256]
            kwargs = {'output_type': record['selected_output']} if family == 'TabPFN' else {}
            outs.append(model.predict(q, **kwargs))
            if start == 0 or start % 1024 == 0:
                print('QUERY', tag, start+len(q), '/', len(queries), 'seconds', round(time.perf_counter()-loaded, 2), flush=True)
        values = np.concatenate(outs).reshape(len(selected), 2**G).astype(float)
        paths = [record['model_path']]
        del model, queries
    else:
        models, scale, paths = load_neural(fold, family)
        np.testing.assert_allclose(scale['mean'], means.to_numpy(), atol=1e-12)
        loaded = time.perf_counter()
        values = []
        for s in selected.itertuples(index=False):
            prefix = data.loc[data.trial.eq(s.trial) & data.minute.le(s.minute)].copy()
            assert len(prefix) == s.prefix_length
            values.append(neural_predict(models, prefix, primary_reference(prefix, means), scale, family))
            if len(values) % 10 == 0:
                print('QUERY', tag, len(values), '/', len(selected), 'seconds', round(time.perf_counter()-loaded, 2), flush=True)
        values = np.stack(values)
        del models
    reference = archived(fold, family).loc[selected.row].to_numpy()
    replay = np.abs(values[:, -1]-reference)
    assert replay.max() < 5e-5, (tag, replay.max())
    phi = exact_phi(values)
    residual = np.abs(phi.sum(-1)-(values[:, -1]-values[:, 0]))
    assert residual.max() < 1e-10 and np.isfinite(values).all()
    np.savez_compressed(result_path, values=values, phi=phi, sample_id=selected.sample_id.to_numpy(), replay=replay)
    save_json(OUT/(tag+'.json'), dict(fold=fold, family=family, samples=len(selected), coalitions=2**G,
        seconds=time.perf_counter()-started, load_and_preparation_seconds=loaded-started,
        replay_max_C=float(replay.max()), additivity_max_C=float(residual.max()),
        model_sha256={p:sha(p) for p in paths}, prediction_sha256=sha(result_path),
        selection_sha256=sha(BASE/'selection.csv'), code_sha256=sha(__file__),
        baseline='training-fold observed channel mean, original observation masks; value-only baseline Shapley'))
    print('DONE', tag, 'seconds', round(time.perf_counter()-started, 2), 'replay', replay.max(), flush=True)
    gc.collect()


def aggregate(selection):
    records, coalitions, ids, checks = [], [], [], []
    for fold in range(5):
        values = []
        for family in MEMBERS:
            tag = f'fold{fold}_{family.replace("-", "")}'
            p = OUT/(tag+'.npz')
            if not p.exists():
                print('INCOMPLETE', p.name, flush=True)
                return False
            a = np.load(p)
            expect = selection.loc[selection.fold.eq(fold)].sample_id.to_numpy()
            np.testing.assert_array_equal(a['sample_id'], expect)
            values.append(a['values'])
            checks.append(json.loads((OUT/(tag+'.json')).read_text(encoding='utf8')))
        v = np.stack(values, axis=-1)
        coalitions.append(v)
        ids.extend(expect.tolist())
        frame = selection.set_index('sample_id').loc[expect]
        for j, model in enumerate(MEMBERS+['Fixed']):
            val = v[:, :, j] if j < 4 else v.mean(-1)
            phi = exact_phi(val)
            for k, sid in enumerate(expect):
                for g, name in enumerate(GROUP_NAMES):
                    row = frame.loc[sid].to_dict()
                    row.update(sample_id=int(sid), model=model, group=name, phi=float(phi[k, g]),
                        absolute_phi=float(abs(phi[k, g])), pred=float(val[k, -1]), baseline=float(val[k, 0]))
                    records.append(row)
        np.testing.assert_allclose(exact_phi(v.mean(-1)), np.mean([exact_phi(v[:, :, j]) for j in range(4)], axis=0), atol=1e-12, rtol=0)
    pd.DataFrame(records).to_csv(OUT/'group_shap.csv', index=False)
    np.savez_compressed(OUT/'coalitions.npz', values=np.concatenate(coalitions), sample_id=np.array(ids),
                        members=np.array(MEMBERS), group_names=np.array(GROUP_NAMES), coalition_bits=COALITIONS)
    save_json(OUT/'verification.json', dict(status='passed', samples=len(ids), participants=selection.person.nunique(),
        models=MEMBERS+['Fixed'], groups=GROUPS, exact_coalitions=2**G, test_subjects_only=True,
        max_replay_C=max(v['replay_max_C'] for v in checks), max_additivity_C=max(v['additivity_max_C'] for v in checks),
        ensemble_linearity='passed with common grouping, reference and Celsius outputs',
        limitations=['Baseline defines an artificial flat training-mean contrast.', 'Masks are held fixed and not attributed.',
        'Coalitions can be off-support; this is model dependence, not a physiological intervention.',
        'One stage-representative endpoint per person, not all-minute population importance.',
        'Conventional-skin group contains 11 channels; group sizes differ.'], components=checks))
    print('ALL GROUP SHAP COMPLETE', flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=range(5))
    p.add_argument('--family', choices=MEMBERS)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--aggregate', action='store_true')
    args = p.parse_args()
    selection = pd.read_csv(BASE/'selection.csv')
    assert len(selection) == 200 and selection.person.nunique() == 40
    if args.aggregate:
        aggregate(selection)
        return
    data = pd.read_pickle(ROOT/'outputs/raw_tcn_all_v1/raw.pkl')
    with threadpool_limits(limits=4):
        for fold in ([args.fold] if args.fold is not None else range(5)):
            assert set(selection.loc[selection.fold.eq(fold), 'person']) == set(SPLITS[fold]['test'])
            for family in ([args.family] if args.family else MEMBERS):
                run_family(fold, family, data, selection, args.benchmark)
    if not args.benchmark:
        aggregate(selection)


if __name__ == '__main__':
    main()
