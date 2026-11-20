# Full-history TCN with 5-minute sparse training on Omius

The TCN uses forehead skin temperature and heart rate. During training, each source trial in each epoch has a 50% chance of retaining only observations spaced at least five minutes apart per channel. The model is selected on PROSPIE validation participants. Five source folds and two seeds contribute equally to the final prediction. Omius is scored at its recorded cadence.

## Running with the archived workspace

Use the original experiment Python environment (PyTorch with CUDA for training). Set `CTF_WORKSPACE` to the separate research workspace containing the original trainer `run_regional_neural_transfer_v2.py`, its modules and the following artifacts under `outputs/`:

- `cpu_baselines_v1/splits.json` and `strict_confirmation_v1/clean_raw.pkl`
- `omius_head2_adapt_20261117/` with its original protocol, five cache manifests and cached inputs
- `external_peer_completion_v3/` with Omius data, checksums, baseline predictions, records and reference summary

These private artifacts are distributed separately. The original manifests contain paths to their caches and checkpoints; those paths must resolve on the replay machine. The archived trainer uses CUDA and its original Windows CPU affinity settings.

From the repository root in PowerShell:

```powershell
$env:CTF_WORKSPACE = 'D:\path\to\original-workspace'
python experiments/omius_sparse_tcn/run_omius_sparse_tcn.py
python experiments/omius_sparse_tcn/evaluate_omius_sparse_tcn.py
```

By default, new fits and scores are written under `CTF_WORKSPACE/outputs/omius_sparse_tcn_reproduction/`. Set `CTF_OUTPUT` to use another directory. To score an existing completed experiment without retraining, set `CTF_OUTPUT` to its output directory and run only the second command.

## Reference results

[Aggregate results](reference_summary.csv) cover 10 participants and 160 paired time points. Overall participant-mean MAE was 1.070658 °C for the sparse-trained TCN, 1.061257 °C for the standard TCN and 0.930966 °C for the standard GRU. The evaluator also reports the two headband conditions separately and checks the standard-model scores against their archived values.
