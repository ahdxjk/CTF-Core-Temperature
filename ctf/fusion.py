"""The fixed, equal-family CTF rule (inputs must be in the same units)."""
import numpy as np


def equal_family_fusion(msca, he_tcn, tabpfn, extra_trees):
    """Average four already aligned family predictions in degrees Celsius.

    Average multiple seed predictions within each neural family before calling.
    """
    values = [np.asarray(x, dtype=float) for x in (msca, he_tcn, tabpfn, extra_trees)]
    if any(x.shape != values[0].shape for x in values):
        raise ValueError('All family predictions must have identical shapes and alignment')
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError('All predictions must be finite')
    return np.stack(values).mean(axis=0)
