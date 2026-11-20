"""Exactly 400 causal input features per physical channel and minute."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from run_raw_tcn_all_v1 import TCN

WINDOWS = (3, 5, 7, 10, 12, 15, 20, 25, 30, 45, 60, 90)
STATS = ('mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'slope_per_min',
         'last_minus_first', 'last_minus_mean', 'coverage', 'observed_span_fraction', 'observations_2plus')
MODELS = ('Shared400', 'Direct400')


def feature_names():
    names = ['last_z', 'observed', 'age_hours', 'ever_observed']
    names += [f'window{w}_{s}' for w in WINDOWS for s in STATS]
    names += [f'lag{k}_{s}' for k in range(1, 61) for s in ('observed_z', 'observed')]
    names += [f'past_block{b}_{s}' for b in range(1, 31) for s in ('mean', 'std', 'slope_per_min', 'coverage')]
    assert len(set(names)) == len(names) == 400
    return names


def trial_features(encoded):
    """One real trial, [minutes, channels, 4] from the frozen training-fold encoder."""
    n, channels, slots = encoded.shape
    assert slots == 4 and n > 0
    result = np.zeros((n, channels, 400), dtype=np.float32)
    result[..., :4] = encoded
    observed = encoded[..., 1].astype(bool)
    frame = pd.DataFrame(np.where(observed, encoded[..., 0], np.nan), dtype=float)
    times = np.broadcast_to(np.arange(n)[:, None], (n, channels))
    stamps = pd.DataFrame(np.where(observed, times, np.nan))
    column_ids = np.arange(channels)[None, :]
    measured = frame.to_numpy()

    def summaries(width, detailed):
        roll = frame.rolling(width, min_periods=1)
        t_roll = stamps.rolling(width, min_periods=1)
        count = roll.count().to_numpy()
        mean = roll.mean().to_numpy()
        std = roll.std(ddof=0).to_numpy()
        sum_t = t_roll.sum().to_numpy()
        sum_tt = stamps.pow(2).rolling(width, min_periods=1).sum().to_numpy()
        sum_tx = (stamps * frame).rolling(width, min_periods=1).sum().to_numpy()
        denominator = sum_tt - sum_t**2 / np.maximum(1, count)
        slope = np.divide(sum_tx - sum_t * mean, denominator, out=np.zeros_like(mean),
                          where=(count >= 2) & (denominator > 0))
        coverage = count / width  # before trial start contributes no observations
        if not detailed:
            return mean, std, slope, coverage
        first_t = t_roll.min().to_numpy()
        last_t = t_roll.max().to_numpy()
        first = measured[np.nan_to_num(first_t).astype(int), column_ids]
        last = measured[np.nan_to_num(last_t).astype(int), column_ids]
        return (mean, std, roll.min(), roll.quantile(.25), roll.median(), roll.quantile(.75),
                roll.max(), slope, np.where(count > 0, last-first, 0),
                np.where(count > 0, last-mean, 0), coverage,
                (last_t-first_t)/(width-1), count >= 2)

    offset = 4
    for width in WINDOWS:
        for values in summaries(width, detailed=True):
            result[..., offset] = np.nan_to_num(np.asarray(values), nan=0.)
            offset += 1
    for lag in range(1, 61):
        if lag < n:
            result[lag:, :, offset] = np.where(observed[:-lag], encoded[:-lag, :, 0], 0)
            result[lag:, :, offset+1] = observed[:-lag]
        offset += 2
    blocks = np.nan_to_num(np.stack(summaries(4, detailed=False), axis=-1), nan=0.)
    for block in range(30):
        lag = 4*block + 1  # block 1 = t-4..t-1, block 30 = t-120..t-117
        if lag < n:
            result[lag:, :, offset:offset+4] = blocks[:-lag]
        offset += 4
    assert offset == 400 and np.isfinite(result).all()
    return result


class ExpandedTCN(nn.Module):
    """Original 64-wide TCN; the added feature path starts at zero."""
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in MODELS:
            raise ValueError(mode)
        self.mode = mode
        self.backbone = TCN(channels*4, 64)
        with torch.random.fork_rng(devices=[]):
            self.extra = nn.Linear(396, 4, bias=False) if mode == 'Shared400' else nn.Linear(channels*396, 64, bias=False)
            nn.init.zeros_(self.extra.weight)

    def forward(self, x):
        base, extra = x[..., :4], x[..., 4:]
        if self.mode == 'Shared400':
            return self.backbone((base + self.extra(extra)).flatten(2))
        initial = self.backbone.embed(base.flatten(2).transpose(1, 2)) + self.extra(extra.flatten(2)).transpose(1, 2)
        return self.backbone.out(self.backbone.blocks(F.silu(initial))).squeeze(1)
