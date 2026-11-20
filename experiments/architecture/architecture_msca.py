"""One shared causal encoder for PROSPIE sensor panels, followed by a scalar Tc head.

Input shape [batch, minute, 22 named slots, 3]: standardized current raw value,
fresh observation mask, and installed-channel mask. Missing readings need no
forward fill. The caller supplies complete trials or their chronological prefixes.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = 22
WIDTH = 88
HEADS = 4
DROPOUT = 0.1
# Groups are contiguous learned features, not anatomical groups of sensors.
KERNEL_GROUPS = {"K3": ((3, 88),), "MK357": ((3, 32), (5, 32), (7, 24))}


class MultikernelBackbone(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in KERNEL_GROUPS:
            raise ValueError(f"Unknown backbone variant: {variant!r}")
        self.variant = variant
        self.width = WIDTH
        self.groups = KERNEL_GROUPS[variant]
        self.sensor_mix = nn.Linear(3 * CHANNELS, WIDTH)
        self.convs = nn.ModuleList(
            nn.Conv1d(size, size, kernel_size=kernel, groups=size)
            for kernel, size in self.groups
        )
        self.token_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.output_projection = nn.Linear(WIDTH, WIDTH)
        self.log_lag_slope = nn.Parameter(torch.zeros(HEADS))
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(2 * WIDTH, WIDTH),
        )
        self.output_norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(DROPOUT)
        self.readout = nn.Linear(WIDTH, 1)

    def encode(self, x):
        """Return one 88-dimensional causal representation per input minute."""
        if x.ndim != 4 or x.shape[2:] != (CHANNELS, 3) or x.size(1) == 0:
            raise ValueError(f"Expected a nonempty [batch, minute, {CHANNELS}, 3] tensor")
        installed = x[..., 2] > 0
        observed = installed & (x[..., 1] > 0)
        value = torch.where(observed, x[..., 0], 0.0)
        slots = torch.stack((value, observed.to(x.dtype), installed.to(x.dtype)), dim=-1)
        features = self.sensor_mix(slots.flatten(2)).transpose(1, 2)

        branches = features.split([size for _, size in self.groups], dim=1)
        local = torch.cat([
            conv(F.pad(branch, (kernel - 1, 0)))
            for branch, (kernel, _), conv in zip(branches, self.groups, self.convs)
        ], dim=1)
        tokens = self.token_norm(F.silu(local.transpose(1, 2)))

        batch, steps = tokens.shape[:2]
        head_size = WIDTH // HEADS
        qkv = self.qkv(tokens).reshape(batch, steps, 3, HEADS, head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(head_size)
        minute = torch.arange(steps, device=x.device)
        lag = minute[:, None] - minute[None, :]
        log_lag = torch.log1p(lag.clamp_min(0).to(tokens.dtype))
        scores = scores + self.log_lag_slope[:, None, None] * log_lag
        weights = scores.masked_fill(lag < 0, -torch.inf).softmax(dim=-1)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, steps, WIDTH)
        attended = self.attention_norm(tokens + self.dropout(self.output_projection(mixed)))
        return self.output_norm(attended + self.dropout(self.feed_forward(attended)))

    def forward(self, x):
        """Estimate current Tc in the training target scale, shape [batch, minute]."""
        return self.readout(self.encode(x)).squeeze(-1)


def build_model(variant):
    return MultikernelBackbone(variant)


def check_structure(model):
    """Verify that the configured stem physically uses its declared kernels."""
    groups = [
        (conv.kernel_size[0], conv.out_channels, conv.groups, conv.padding, conv.dilation)
        for conv in model.convs
    ]
    expected = [(k, size, size, (0,), (1,)) for k, size in KERNEL_GROUPS[model.variant]]
    if groups != expected or sum(size for _, size in model.groups) != WIDTH:
        raise AssertionError(f"Unexpected causal stem: {groups}")
    return {"kernels": list(model.groups), "parameters": sum(p.numel() for p in model.parameters())}
