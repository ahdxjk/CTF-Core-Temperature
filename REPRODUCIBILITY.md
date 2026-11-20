# Reproducibility status

This is a pre-publication code preparation folder. The architecture-only synthetic example is runnable; the full paper pipeline is not yet a self-contained reproduction.

The architecture ablation runner requires the exact `clean_raw.pkl` (SHA-256 `0e6a48308ac45f4fa510a7e9c254a943648488811bf739abbd6619af1edd5495`) and archived participant splits. They are not included. It requires 17,264 timeline rows and 17,153 retained target values; masking spreadsheet formulas must occur before feature extraction and training. `clean_raw.pkl` contains participant-level records and must not be placed in a public repository without confirming the applicable terms and any necessary approvals.

The forecasting and external-adaptation snapshots depend on additional original workspace modules, datasets, participant splits, per-fold predictions, evaluation scripts and model checkpoints. They include Windows-specific control code and archived paths and are not standalone public workflows. Their presence documents the original implementation, not completion of a reproducibility release. In particular, nominal activity-plan availability must be audited at each prediction origin; do not construct model inputs from realized future actions.

The complete CTF also requires the TabPFN 3.5 pretrained model and its separately licensed weights, fold-specific fitted contexts, ExtraTrees models and training-only normalization. These are not distributed here. Explain any differences in software versions, preprocessing, model selection and evaluation support when comparing results.

Before publication, audit ownership and licenses of every included script, select a repository license, document data-preparation commands and permitted access, replace reference snapshots with portable pipelines, and release validated model checkpoints or a complete reconstruction recipe as allowed. A numerical parity run against archived fold predictions is still required.
