"""Architecture-only synthetic smoke check; not a trained estimator."""
import numpy as np
import torch
from ctf import MultikernelBackbone, ExpandedTCN, trial_features
from ctf.forecast import DirectMSCA, DirectHE

torch.manual_seed(42)
torch.set_num_threads(1)
T, C = 16, 2
slots = torch.zeros(1, T, 22, 3)
slots[:, :, :C, 0] = torch.randn(1, T, C)
slots[:, :, :C, 1:] = 1
raw = np.zeros((T, C, 4), dtype=np.float32)
raw[:, :, 0] = slots[0, :, :C, 0].numpy()
raw[:, :, 1] = 1
raw[:, :, 3] = 1
hist = torch.from_numpy(trial_features(raw))[None]
context = torch.zeros(1, T, 2)
models = [MultikernelBackbone('MK357'), ExpandedTCN(C, 'Direct400'), DirectMSCA(), DirectHE(C)]
for m in models:
    m.eval()
with torch.no_grad():
    outputs = [models[0](slots), models[1](hist), models[2](slots, context), models[3](hist, context)]
    assert all(y.shape == (1, T) and torch.isfinite(y).all() for y in outputs)
    # Predictions at earlier time steps must not depend on future input changes.
    changed = slots.clone(); changed[:, 10:, :, 0] += 20
    torch.testing.assert_close(models[0](slots)[:, :10], models[0](changed)[:, :10], rtol=1e-5, atol=1e-5)
print('Synthetic architecture/causality checks passed. Weights are random; outputs are not temperatures.')
