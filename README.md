# Complementary Temporal Fusion for Core Temperature

Research code associated with a manuscript on causal five-phase analysis, non-invasive core-temperature estimation and direct multi-horizon forecasting. The four-member CTF combines MSCA-Net, HE-TCN, TabPFN and ExtraTrees; neural seed predictions are averaged within family and family predictions are equally weighted.

## What is included

- [Omius Sparse5 TCN](experiments/omius_sparse_tcn/README.md) includes the two-channel training and external scoring scripts and the aggregate reference results.
- `ctf/msca.py`: archived multiscale causal-attention architecture.
- `ctf/history.py`: 400 causal descriptors per active channel and the HE-TCN architecture.
- `ctf/fusion.py`: fixed equal-weight fusion of four aligned family predictions.
- `ctf/forecast.py`: direct prediction heads with configurable origin-available activity context. Separate models must be trained for each horizon and condition.
- `experiments/architecture/`: portable source for the matched K3/MK357 and State4/HE400 architecture ablations.
- `research_reference/`: preserved research scripts for activity context, direct forecasting, fusion, external cadence adaptation and attribution. They are **reference snapshots, not portable entry points**: they expect the original workspace, data, checkpoints, other dependencies and precomputed results. Paths identifying the original computer were redacted. This folder does not yet reproduce all manuscript tables end-to-end.
- `examples/synthetic_demo.py`: random-data architecture and causality check, not a trained model or temperature prediction.

## Run the architecture check

Install Python 3.12 and PyTorch suitable for your OS and CPU/GPU. The versions in `requirements.txt` are the locally tested versions, not a cross-platform lock file. From this directory:

```bash
python -m pip install -r requirements.txt
python -m examples.synthetic_demo
```

The example uses synthetic data and randomly initialized weights; it does not reproduce the paper's performance. No pretrained CTF, TabPFN model, fold splits or patient records are included.

## Real data and reproducibility

The PROSPIE data are distributed separately by Loughborough University: https://doi.org/10.17028/rd.lboro.26076577.v1. The dataset license is CC BY-NC 4.0; credit the creators and check restrictions independently. The research code and third-party datasets and pretrained models each have their own terms. Do not infer the code license from the dataset license.

The archived experiments masked formula-derived input/target cells before preprocessing and used participant-disjoint train/validation/test splits. Merely downloading the workbook does not create the audited `clean_raw.pkl`, phase labels, splits, causal context, pretrained weights or predictions required by the ablation scripts. Obtain and validate these artifacts separately; `run_architecture.py` enforces the archived clean-data hash. See `REPRODUCIBILITY.md` for the missing steps and scope.

## Code license

A code license has not yet been selected by the authors. Do not assume this repository grants an open-source license until the rights holders choose and add one. No third-party code or model weights are relicensed here.
