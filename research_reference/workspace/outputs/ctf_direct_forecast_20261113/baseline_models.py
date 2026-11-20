"""Independent causal forecasting controls with matched raw sensor information.

The runner owns folds, normalization, targets, losses, seeds and early stopping.
Both models consume the complete trial prefix; no fixed-length window is taken.
Sensor x is [batch, minute, 14, 4]: last/fresh-mask/age-hours/ever-observed.
Activity is [batch, minute, 2]: current work indicator and known indicator.
Outputs are [batch, minute] in the runner's normalized future-target scale.
"""

import torch
from torch import nn
from run_raw_tcn_all_v1 import TCN


def _inputs(x, activity):
    if x.ndim != 4 or x.shape[2:] != (14, 4) or x.shape[1] == 0:
        raise ValueError("Expected a nonempty [batch, minute, 14, 4] sensor tensor")
    if activity.shape != (*x.shape[:2], 2):
        raise ValueError("Expected activity [batch, minute, 2] matching the sensors")
    return torch.cat((x.flatten(2), activity), dim=-1)


class GRUForecast(nn.Module):
    """Two-layer unidirectional GRU, width 64, direct horizon-specific head."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(58, 64, num_layers=2, batch_first=True,
                          dropout=0.1, bidirectional=False)
        self.readout = nn.Linear(64, 1)

    def forward(self, x, activity):
        representation, _ = self.gru(_inputs(x, activity))
        return self.readout(representation).squeeze(-1)


class TCNForecast(nn.Module):
    """Original 64-wide causal TCN, without the HE 396-feature extension.

    Eight residual blocks, two depthwise-k3/pointwise convolutions per block,
    dilations 1..128 and receptive field 1021 minutes, as in the source TCN.
    """

    def __init__(self):
        super().__init__()
        self.network = TCN(58, 64)

    def forward(self, x, activity):
        return self.network(_inputs(x, activity))


def build(kind):
    if kind == "GRU":
        return GRUForecast()
    if kind == "TCN":
        return TCNForecast()
    raise ValueError(f"Unknown independent forecasting control: {kind!r}")
