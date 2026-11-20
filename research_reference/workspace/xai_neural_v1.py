"""Frozen MSCA value attribution on prespecified participant/phase prefixes.

No training or test-selected hyperparameters. The flat reference is an artificial
training-distribution reference, not an observed physiological intervention.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from multikernel_backbone_model_v1 import build_model
from prepare_multikernel_backbone_v1 import CHANNELS, SEEDS, SPLITS, load_data, MODEL_DIR

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/interpretability_v1/neural'
SHARED = OUT.parent
LAG_BINS = ((0, 4, '0-4'), (5, 14, '5-14'), (15, 29, '15-29'),
            (30, 59, '30-59'), (60, 10000, '60+'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG_SEED = 20261106


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == value, f'Frozen protocol differs: {path}'
    else:
        dump(path, value)


class FrozenMSCA:
    def __init__(self, fold):
        self.fold = int(fold)
        self.models = []
        self.hashes = {}
        for seed in SEEDS:
            path = MODEL_DIR / f'MK357_s{seed}_f{fold}.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['channels'] == CHANNELS
            assert ck['split'] == SPLITS[fold]
            model = build_model('MK357').to(DEVICE)
            model.load_state_dict(ck['state'])
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.models.append(model)
            self.hashes[str(seed)] = sha(path)
            if len(self.models) == 1:
                self.scale = ck['preprocessing']
            else:
                assert self.scale == ck['preprocessing']

    def encode(self, frame):
        values = frame[CHANNELS].to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        z = (values - np.asarray(self.scale['mean'])) / np.asarray(self.scale['std'])
        return np.stack((np.where(observed, z, 0), observed, np.ones_like(values)), axis=-1).astype(np.float32)

    def output(self, x):
        return torch.stack([m(x)[:, -1] for m in self.models]).mean(0) * self.scale['ystd'] + self.scale['ymean']

    def predict(self, inputs, batch=32):
        inputs = np.asarray(inputs)
        if inputs.ndim == 3:
            inputs = inputs[None]
        with torch.no_grad():
            return np.concatenate([self.output(torch.as_tensor(a, device=DEVICE)).cpu().numpy()
                                   for a in np.array_split(inputs, max(1, int(np.ceil(len(inputs) / batch))))])

    def ig(self, x, reference, maximum=256):
        # Gauss-Legendre integrates the input-gradient path over alpha in [0, 1].
        delta = torch.as_tensor(x - reference, device=DEVICE)
        base = torch.as_tensor(reference, device=DEVICE)
        predictions = self.predict(np.stack((x, reference)))
        target_delta = float(predictions[0] - predictions[1])
        for steps in (32, 64, 128, 256):
            nodes, weights = np.polynomial.legendre.leggauss(steps)
            nodes = (nodes + 1) / 2
            weights = weights / 2
            integral = torch.zeros_like(base)
            for k in range(0, steps, 16):
                a = torch.as_tensor(nodes[k:k+16], device=DEVICE, dtype=torch.float32)[:, None, None, None]
                interpolated = (base[None] + a * delta[None]).requires_grad_(True)
                grad = torch.autograd.grad(self.output(interpolated).sum(), interpolated)[0]
                w = torch.as_tensor(weights[k:k+16], device=DEVICE, dtype=torch.float32)[:, None, None, None]
                integral += (grad * w).sum(0)
            attribution = (delta * integral).detach().cpu().numpy()[..., 0]
            error = float(attribution.sum() - target_delta)
            tolerance = max(.001, .01 * abs(target_delta))
            if abs(error) <= tolerance or steps >= maximum:
                break
        return attribution, dict(pred=float(predictions[0]), reference_pred=float(predictions[1]),
                                 ig_sum=float(attribution.sum()), completeness_error=error,
                                 ig_tolerance=tolerance, ig_steps=steps, converged=bool(abs(error) <= tolerance))

    def gradcam(self, x):
        # Hook the post-convolution normalization, before attention. Temporal
        # feature channels already mix sensors and cannot identify body sites.
        maps = []
        for seed, model in zip(SEEDS, self.models):
            captured = []
            hook = model.token_norm.register_forward_hook(lambda module, inputs, output: captured.append(output))
            inp = torch.as_tensor(x[None], device=DEVICE).requires_grad_(True)
            output = model(inp)[:, -1] * self.scale['ystd'] + self.scale['ymean']
            features = captured[0]
            grad = torch.autograd.grad(output.sum(), features)[0]
            cam = (grad.mean(1, keepdim=True) * features).sum(-1)[0].detach().cpu().numpy()
            hook.remove()
            maps.append((seed, cam))
        return maps


def metadata(row):
    return {k: (int(row[k]) if k in ('sample_id', 'fold', 'person', 'trial', 'row') else row[k])
            for k in ('sample_id', 'fold', 'person', 'phase', 'trial', 'row', 'y')}


def flat_reference(x, z):
    ref = x.copy()
    ref[..., 0] = np.where(x[..., 1] > 0, z, 0)
    return ref


def donor_reference(data, split, row, x, model):
    # Training-only, same target phase and sufficient prefix length. Select
    # deterministically by trial/time order, without labels or model residuals.
    candidates = data.loc[data.Participant.isin(split['train']) & data.phase_5.eq(row.phase)
                          & data.minute.ge(len(x)-1)].sort_values(['trial', 'minute', 'source_excel_row'])
    phase_matched = True
    if candidates.empty:
        candidates = data.loc[data.Participant.isin(split['train']) & data.minute.ge(len(x)-1)].sort_values(['trial', 'minute', 'source_excel_row'])
        phase_matched = False
    # A held-out trial can be longer than every training trial. Retain it;
    # record the unsupported leading minutes instead of dropping the case.
    if candidates.empty:
        training = data.loc[data.Participant.isin(split['train'])]
        candidates = training.loc[training.minute.eq(training.minute.max())].sort_values(['trial', 'minute', 'source_excel_row'])
        phase_matched = False
    chosen = candidates.iloc[(len(candidates)-1)//2]
    prefix = data.loc[data.trial.eq(chosen.trial) & data.minute.le(chosen.minute)].sort_values('minute')
    # Causal fill within the donor's history, then align the trailing prefix.
    values = prefix[CHANNELS].ffill().tail(len(x))
    padding = len(x) - len(values)
    assert padding >= 0
    filled = values.notna().to_numpy()
    z = ((values - model.scale['mean']) / model.scale['std']).fillna(0).to_numpy(np.float32)
    if padding:
        z = np.pad(z, ((padding, 0), (0, 0)))
        filled = np.pad(filled, ((padding, 0), (0, 0)), constant_values=False)
    ref = flat_reference(x, z)
    covered = (filled & (x[..., 1] > 0)).sum() / max(1, (x[..., 1] > 0).sum())
    details = dict(donor_person=int(chosen.Participant), donor_trial=int(chosen.trial),
                   donor_minute=float(chosen.minute), donor_phase=str(chosen.phase_5),
                   donor_same_phase=phase_matched, donor_filled_value_coverage=float(covered),
                   donor_padding_minutes=padding)
    return ref, details


def occlude(model, x, ref, attribution, row, reference, rank_source):
    n = len(x)
    lag = n - 1 - np.arange(n)
    block = lag // 5
    nb = int(block.max()) + 1
    score = np.array([np.abs(attribution[block == b]).sum() for b in range(nb)])
    top = np.argsort(-score, kind='stable')
    least = np.argsort(score, kind='stable')
    rng = np.random.default_rng(RNG_SEED + int(row.sample_id) * 73)
    random_orders = [rng.permutation(nb) for _ in range(5)]
    inputs, details = [], []
    pred = float(model.predict(x)[0])
    for fraction in (.1, .3, .5):
        k = max(1, int(np.ceil(nb * fraction)))
        for label, orders in (('top', [top]), ('least', [least]), ('random', random_orders)):
            for repeat, order in enumerate(orders):
                mask = np.isin(block, order[:k])
                modified = x.copy()
                modified[mask, :, 0] = ref[mask, :, 0]
                inputs.append(modified)
                details.append(dict(**metadata(row), reference=reference, rank_source=rank_source,
                                    order=label, fraction=fraction, repeat=repeat,
                                    blocks=k, total_blocks=nb, actual_fraction=k/nb, replaced_minutes=int(mask.sum()),
                                    replaced_observed_cells=int(x[mask, :, 1].sum()), pred=pred))
    perturb = model.predict(np.stack(inputs))
    for item, yp in zip(details, perturb):
        item.update(perturbed_pred=float(yp), signed_change_C=float(yp-pred), abs_change_C=float(abs(yp-pred)),
                    delta_absolute_error_C=float(abs(yp-row.y)-abs(pred-row.y)))
    return details


def protocol():
    return dict(task='Current rectal-temperature estimate; frozen All22 MSCA two-seed mean only, not the four-family ensemble',
                sample_selection_sha256=sha(SHARED/'selection.csv'),
                model_source_sha256=sha(ROOT/'multikernel_backbone_model_v1.py'),
                script_sha256=sha(__file__), device=str(DEVICE), seed=RNG_SEED,
                selection='Read shared 200 person-phase median targets; no residual or attribution-based selection',
                reference='Primary train-fold channel observed mean (z=0); secondary observed q25, both preserve all binary masks',
                reference_caution='Flat artificial trajectories are model reference stress tests, not physiological counterfactuals',
                quadrature='Gauss-Legendre 32,64,128,256; stop absolute completeness error<=max(0.001 C,1% absolute output difference)',
                output='Celsius; average two seed outputs before taking absolute attribution',
                lag_bins=[b[2] for b in LAG_BINS], unavailable_lags='NaN with available_minutes=0',
                fidelity='5-minute blocks backwards from target; sum absolute cell IG ranks; replace all observed sensor values, keep masks; budgets10/30/50%; top/least/5 random orders',
                donor='Each participant P3 (fallback first phase); same primary-mean IG ranking; training-only same-phase length-matched donor candidate median by trial/time, causal donor ffill, remaining missing train mean; target masks fixed. If no training prefix is sufficiently long, use a longest training prefix and explicitly record leading training-mean padding.',
                amendment='A first incomplete execution stopped at sample7 because test prefix134 exceeds fold0 training maximum133; no attribution results were inspected to change this availability-only fallback.',
                sanity='Lowest participant id P3 per fold (fallback first phase); output-readout randomized then all modules randomized, fixed RNG; mean reference IG32..256',
                gradcam='Per-seed signed and positive temporal map at token_norm output. Not channel attribution or complete output decomposition',
                uncertainty='Participant unit; explanations on selected targets only; repeatedly explored existing splits; no confirmatory or physiological causal claims')


def run(folds=None, limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = pd.read_csv(SHARED/'selection.csv')
    data = load_data()
    if limit is None:
        freeze(OUT/'protocol.json', protocol())
    all_metrics, all_channels, all_lags, all_occlusion, all_cam, all_sanity, all_donors = [], [], [], [], [], [], []
    arrays = {}
    start = time.perf_counter()
    for split in SPLITS:
        fold = split['fold']
        if folds is not None and fold not in folds:
            continue
        model = FrozenMSCA(fold)
        chosen = selection[selection.fold.eq(fold)]
        if limit:
            chosen = chosen.head(limit)
        train = data[data.Participant.isin(split['train'])]
        q25 = ((train[CHANNELS].quantile(.25).to_numpy()-model.scale['mean']) / model.scale['std']).astype(np.float32)
        archived = []
        for seed in SEEDS:
            p = ROOT / f'outputs/multikernel_backbone_v1/fits/MK357_s{seed}_f{fold}/test.csv.gz'
            archived.append(pd.read_csv(p).query("panel == 'All22'").set_index('row').pred)
        archived = pd.concat(archived, axis=1)
        donor_ids = set()
        for person, group in chosen.groupby('person'):
            preferred = group[group.phase.eq('P3')]
            donor_ids.add(int((preferred if len(preferred) else group.sort_values('phase')).iloc[0].sample_id))
        sanity_id = min(donor_ids, key=lambda sid: int(chosen.loc[chosen.sample_id.eq(sid), 'person'].iloc[0]))
        for _, row in chosen.iterrows():
            tick = time.perf_counter()
            prefix = data.loc[data.trial.eq(row.trial) & data.minute.le(row.minute)].sort_values('minute')
            assert len(prefix) == row.prefix_length and prefix.iloc[-1].source_excel_row == row.row
            assert prefix.Participant.nunique() == 1 and int(prefix.Participant.iloc[0]) in split['test']
            # CSV round-tripping can differ by one float64 ULP.
            assert np.isclose(prefix.iloc[-1].Corerectal, row.y, atol=1e-10, rtol=0)
            x = model.encode(prefix)
            pred = float(model.predict(x)[0])
            replay = float(abs(pred - archived.loc[int(row.row)].mean()))
            assert replay < 5e-5, (row.sample_id, replay)
            with torch.no_grad():
                tx = torch.as_tensor(x[None], device=DEVICE)
                individual = [float((net(tx)[0, -1]*model.scale['ystd']+model.scale['ymean']).cpu()) for net in model.models]
            seed_replay = float(np.max(np.abs(np.asarray(individual)-archived.loc[int(row.row)].to_numpy())))
            assert seed_replay < 5e-5, (row.sample_id, seed_replay)
            causal_error = 0.
            if int(row.sample_id) == sanity_id:
                future = np.repeat(x[-1:], 7, axis=0)
                future[..., 0] += np.random.default_rng(RNG_SEED+fold).normal(size=future[..., 0].shape).astype(np.float32)
                extended = torch.as_tensor(np.concatenate([x, future])[None], device=DEVICE)
                with torch.no_grad():
                    for net in model.models:
                        before = net(tx)[0]*model.scale['ystd']+model.scale['ymean']
                        after = net(extended)[0, :len(x)]*model.scale['ystd']+model.scale['ymean']
                        causal_error = max(causal_error, float((before-after).abs().max().cpu()))
                assert causal_error < 5e-5
            mean_attr = None
            for reference, z in (('train_mean', np.zeros(len(CHANNELS))), ('train_q25', q25)):
                ref = flat_reference(x, z)
                attribution, metrics = model.ig(x, ref)
                missing_max = float(np.max(np.abs(attribution[x[..., 1] == 0]))) if (x[..., 1] == 0).any() else 0.
                assert missing_max == 0
                metrics.update(**metadata(row), reference=reference, replay_max_diff=replay,
                               missing_abs_attr_max=missing_max, prefix_length=len(x),
                               per_seed_replay_max_diff=seed_replay,
                               causal_checked=bool(int(row.sample_id)==sanity_id), causal_prefix_max_diff=causal_error)
                all_metrics.append(metrics)
                arrays[f's{int(row.sample_id)}_{reference}'] = attribution
                for j, channel in enumerate(CHANNELS):
                    all_channels.append(dict(**metadata(row), reference=reference, channel=channel,
                                             signed_C=float(attribution[:, j].sum()), abs_C=float(np.abs(attribution[:, j]).sum()),
                                             observed_minutes=int(x[:, j, 1].sum())))
                lag = len(x)-1-np.arange(len(x))
                for lo, hi, name in LAG_BINS:
                    mask = (lag >= lo) & (lag <= hi)
                    all_lags.append(dict(**metadata(row), reference=reference, lag_bin=name,
                                         signed_C=float(attribution[mask].sum()) if mask.any() else np.nan,
                                         abs_C=float(np.abs(attribution[mask]).sum()) if mask.any() else np.nan,
                                         available_minutes=int(mask.sum()), observed_cells=int(x[mask, :, 1].sum())))
                all_occlusion.extend(occlude(model, x, ref, attribution, row, reference, reference))
                if reference == 'train_mean':
                    mean_attr = attribution
            arrays[f's{int(row.sample_id)}_input'] = x
            arrays[f's{int(row.sample_id)}_minutes'] = prefix.minute.to_numpy()
            for seed, cam in model.gradcam(x):
                for t, value in enumerate(cam):
                    all_cam.append(dict(**metadata(row), seed=seed, input_minute=float(prefix.minute.iloc[t]),
                                        lag_minute=len(x)-1-t, signed_cam=float(value), positive_cam=float(max(0, value))))
            if int(row.sample_id) in donor_ids:
                ref, details = donor_reference(data, split, row, x, model)
                all_donors.append(dict(**metadata(row), **details))
                arrays[f's{int(row.sample_id)}_donor_input'] = ref
                all_occlusion.extend(occlude(model, x, ref, mean_attr, row, 'training_donor', 'train_mean'))
            if int(row.sample_id) == sanity_id:
                states = [{k: v.clone() for k, v in net.state_dict().items()} for net in model.models]
                for mode in ('readout_randomized', 'all_parameters_randomized'):
                    torch.manual_seed(RNG_SEED + fold * 31)
                    for net in model.models:
                        if mode == 'readout_randomized':
                            net.readout.reset_parameters()
                        else:
                            for module in net.modules():
                                if hasattr(module, 'reset_parameters'):
                                    module.reset_parameters()
                            with torch.no_grad():
                                net.log_lag_slope.normal_()
                    attr, check = model.ig(x, flat_reference(x, np.zeros(len(CHANNELS))))
                    valid = x[..., 1] > 0
                    rho = float(spearmanr(mean_attr[valid], attr[valid]).statistic)
                    rho_abs = float(spearmanr(np.abs(mean_attr[valid]), np.abs(attr[valid])).statistic)
                    all_sanity.append(dict(**metadata(row), mode=mode, signed_spearman=rho,
                                           absolute_spearman=rho_abs, **check))
                    for net, state in zip(model.models, states):
                        net.load_state_dict(state)
            print(f"sample={row.sample_id} fold={fold} person={row.person} phase={row.phase} T={len(x)} seconds={time.perf_counter()-tick:.2f}", flush=True)
        # Checkpoint completed folds without discarding prior observations.
        payload = dict(metrics=all_metrics, channels=all_channels, lags=all_lags, occlusion=all_occlusion,
                       cam=all_cam, sanity=all_sanity, donors=all_donors)
        save_results(payload, arrays, 'benchmark_' if limit else '')
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    if limit is None:
        table = pd.DataFrame(all_metrics)
        completion = dict(status='complete', samples=int(table.sample_id.nunique()), people=int(table.person.nunique()),
                          folds=sorted(map(int, table.fold.unique())), explanations=len(table),
                          max_replay_diff=float(table.replay_max_diff.max()), max_completeness_error=float(table.completeness_error.abs().max()),
                          unconverged=int((~table.converged).sum()), nonzero_missing_attribution=int((table.missing_abs_attr_max>0).sum()),
                          donor_samples=len(all_donors), sanity_tests=len(all_sanity), seconds=time.perf_counter()-start)
        dump(OUT/'completion.json', completion)
        print(json.dumps(completion), flush=True)
    else:
        print('BENCHMARK_SECONDS', time.perf_counter()-start, flush=True)


def save_results(p, arrays, prefix):
    names = dict(metrics='sample_metrics', channels='channel_attribution', lags='lag_attribution',
                 occlusion='occlusion', cam='gradcam', sanity='randomization', donors='donors')
    for key, name in names.items():
        pd.DataFrame(p[key]).to_csv(OUT/f'{prefix}{name}.csv', index=False)
    np.savez_compressed(OUT/f'{prefix}cell_attribution.npz', **arrays)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=int)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    run(None if args.fold is None else [args.fold], args.benchmark)
