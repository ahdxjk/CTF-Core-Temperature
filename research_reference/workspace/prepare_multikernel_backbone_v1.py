"""Frozen data contract for the shared, multi-panel PROSPIE backbone."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_raw_tcn_all_v1 import BASE, PHYS, encoding

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/multikernel_backbone_v1'
MODEL_DIR = Path('REPLACE_WITH_MODEL_DIR/multikernel_backbone_v1')
RAW = ROOT / 'outputs/raw_tcn_all_v1/raw.pkl'
SPLIT_FILE = ROOT / 'outputs/cpu_baselines_v1/splits.json'
SPLITS = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))
SEEDS = [20261103, 20261104]
VARIANTS = ['K3', 'MK357']
CHANNELS = PHYS + ['Tfront', 'Rhfront', 'Tback', 'Rhback', 'Temp', 'Humidity', 'Wind']
PANELS = {
    'HeadHR2': ['head', 'HR'],
    'Neck2': ['Ins skin', 'Tamb'],
    'ESWA4': ['head', 'HR', 'Temp', 'Humidity'],
    'Base12': BASE,
    'Neck14': BASE + ['Ins skin', 'Tamb'],
    'NeckBR15': PHYS,
    'All22': CHANNELS,
}
PANEL_ENABLED = torch.tensor([[name in sensors for name in CHANNELS] for sensors in PANELS.values()])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    data = pd.read_pickle(RAW)
    assert data.index.equals(pd.RangeIndex(17264)) and data.source_excel_row.is_unique
    assert data.Corerectal.notna().sum() == 17162 and data.trial.nunique() == 155
    assert data.groupby('trial').Participant.nunique().eq(1).all()
    assert data.groupby('trial').minute.diff().dropna().eq(1).all()
    assert sorted(len(s['test']) for s in SPLITS) == [8] * 5
    assert sorted(p for s in SPLITS for p in s['test']) == sorted(data.Participant.unique())
    for s in SPLITS:
        assert [len(s[k]) for k in ('train', 'validation', 'test')] == [24, 8, 8]
        assert len(set(s['train'] + s['validation'] + s['test'])) == 40
    return data


def pack_fold(data, split):
    x, y, valid, rows, _, groups, weights, scale, _ = encoding(data, split, CHANNELS)
    assert scale['columns'] == CHANNELS
    slots = x.reshape(*x.shape[:2], len(CHANNELS), 4)
    observed = slots[..., 1]
    obs = np.stack((np.where(observed != 0, slots[..., 0], 0), observed), axis=-1)
    assert (rows >= 0).sum() == 17264 and valid.sum() == 17162
    assert np.all(weights[~valid] == 0) and np.isclose(weights.sum(), 1)
    return obs, y, rows, groups, weights, scale


def build_inputs(obs, panel_indices):
    """Force inaccessible values and measurement masks to zero even if the source has data."""
    enabled = PANEL_ENABLED.to(obs.device)[panel_indices].unsqueeze(1).expand(-1, obs.shape[1], -1)
    measured = enabled & obs[..., 1].bool()
    values = torch.where(measured, obs[..., 0], 0.)
    return torch.stack((values, measured.to(obs.dtype), enabled.to(obs.dtype)), dim=-1)


def protocol():
    files = [Path(__file__), ROOT/'multikernel_backbone_model_v1.py', ROOT/'run_multikernel_backbone_v1.py',
             ROOT/'run_raw_tcn_all_v1.py', SPLIT_FILE, RAW]
    return {
        'hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}, 'splits': SPLITS,
        'task': 'Current rectal reference only', 'channels': CHANNELS, 'panels': PANELS,
        'panel_order': list(PANELS), 'variants': VARIANTS, 'seeds': SEEDS,
        'inputs': 'Measured fold-standardised value, measured now, panel enabled. No phase, real Tc or future observations.',
        'preprocessing': 'Sensor and target statistics fit only on training participants. Trial state resets.',
        'training': 'One panel per whole trial per epoch, each trial sees every panel in a seven-epoch cycle. Pair schedule across architectures.',
        'optimizer': 'AdamW lr .001 wd .001, gradient norm cap 1, whole-trial batch 8, max 150 epochs, patience 20.',
        'loss': 'Participant-macro absolute error. No stage optimization or distillation in this backbone comparison.',
        'selection': 'Early stopping by mean of 7 validation panel person-MAEs. No test selection. Report both seeds and fixed mean.',
        'comparison': 'Single shared 22-slot backbone per fit. Identical downstream full-history causal attention and readout. Compare one 3-minute convolution with parallel true 3/5/7-minute kernels.',
        'evaluation': '7 panels, 40 people, 155 trials, 17264 input times, 17162 scored rows/panel. 2 seeds x 5 folds x 2 variants = 20 fits.',
        'limitations': 'Repeatedly explored splits; panel views are paired, not new independent subjects; inference over physical sensors present in the panel.',
    }


def assert_frozen():
    assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8')) == protocol(), 'Frozen protocol mismatch'
    assert json.loads((OUT/'preflight.json').read_text(encoding='utf-8'))['status'] == 'passed'


def preflight():
    from multikernel_backbone_model_v1 import build_model
    torch.manual_seed(20261103)
    torch.set_num_threads(2)
    data = load_data()
    obs = torch.from_numpy(pack_fold(data, SPLITS[0])[0][:7])
    indices = torch.arange(len(PANELS))
    sample = build_inputs(obs, indices)
    assert sample.shape == (7, 141, 22, 3)
    scores = {}
    for i, names in enumerate(PANELS.values()):
        available = torch.tensor([s in names for s in CHANNELS])
        assert torch.equal(sample[i, 0, :, 2].bool(), available)
        changed = obs.clone()
        changed[i, :, ~available] = 100
        assert torch.equal(build_inputs(changed, indices), sample)
    for variant in VARIANTS:
        model = build_model(variant).eval()
        with torch.no_grad():
            original = model(sample)
            assert original.shape == sample.shape[:2] and torch.isfinite(original).all()
            changed = sample.clone()
            changed[:, 55:] = torch.randn_like(changed[:, 55:])
            future = (model(changed)[:, :55] - original[:, :55]).abs().max().item()
            prefix = (model(sample[:, :55]) - original[:, :55]).abs().max().item()
            assert max(future, prefix) < 5e-5
        model.train()
        model(sample).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        scores[variant] = dict(parameters=sum(p.numel() for p in model.parameters()), future=future, prefix=prefix)
    frozen = protocol()
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8')) == frozen
    else:
        save_json(OUT/'protocol.json', frozen)
    save_json(OUT/'preflight.json', dict(status='passed', checks=scores, hidden_sensor_perturbation='passed'))
    print('Preflight passed', scores, flush=True)


if __name__ == '__main__':
    preflight()
