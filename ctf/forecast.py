"""Direct-horizon heads using the archived encoders and origin-available context.

Create a separate instance for each horizon/condition and fit it on that task.
All inputs must be normalized with training-fold statistics.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .msca import MultikernelBackbone
from .history import ExpandedTCN


class DirectMSCA(nn.Module):
    def __init__(self, context_dim=2):
        super().__init__()
        self.encoder = MultikernelBackbone("MK357")
        self.encoder.readout = nn.Identity()
        self.readout = nn.Linear(88 + context_dim, 1)

    def forward(self, x, context):
        return self.readout(torch.cat((self.encoder.encode(x), context), dim=-1)).squeeze(-1)


class DirectHE(nn.Module):
    def __init__(self, channels=14, context_dim=2):
        super().__init__()
        self.encoder = ExpandedTCN(channels, "Direct400")
        self.encoder.backbone.out = nn.Identity()
        self.readout = nn.Linear(64 + context_dim, 1)

    def forward(self, x, context):
        base, extra = x[..., :4], x[..., 4:]
        z = self.encoder.backbone.embed(base.flatten(2).transpose(1, 2))
        z = z + self.encoder.extra(extra.flatten(2)).transpose(1, 2)
        z = self.encoder.backbone.blocks(F.silu(z)).transpose(1, 2)
        return self.readout(torch.cat((z, context), dim=-1)).squeeze(-1)
