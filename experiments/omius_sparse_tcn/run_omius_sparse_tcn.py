"""Fit the two-channel full-history TCN with the archived Sparse5 training policy."""
from pathlib import Path
import os
import sys
import pandas as pd
# The frozen research workspace supplies the trainer, five source splits and caches.
workspace = os.environ.get('CTF_WORKSPACE')
if not workspace:
    raise SystemExit("Set CTF_WORKSPACE to the archived experiment workspace (see README.md).")
ROOT = Path(workspace).expanduser().resolve()
if not (ROOT / 'run_regional_neural_transfer_v2.py').is_file():
    raise SystemExit(f"Archived trainer missing under {ROOT}")
sys.path.append(str(ROOT))
import run_regional_neural_transfer_v2 as r
from run_raw_tcn_all_v1 import TCN

OUT = Path(os.environ.get('CTF_OUTPUT', ROOT / 'outputs/omius_sparse_tcn_reproduction')).resolve()
CACHE = ROOT / 'outputs/omius_head2_adapt_20261117'
SETTING = 'TCN_Sparse5'


def flatten_inputs(raw, kind):
    assert kind == 'Raw4' and raw.shape[-2:] == (2, 4)
    return raw.reshape(*raw.shape[:2], 8)


def main():
    r.OUT, r.MOD, r.COLS = OUT, OUT / 'models', ['head', 'HR']
    r.runtime('cuda')
    r.build_setting = lambda setting: TCN(8, 64)
    r.input_kind = lambda setting: 'Raw4'
    r.input_from_raw = flatten_inputs
    r.structure = lambda setting, model: dict(setting=setting, input_kind='Raw4', width=64,
                                              parameters=sum(p.numel() for p in model.parameters()))
    old = r.read(CACHE/'protocol.json')
    source = ROOT/'outputs/strict_confirmation_v1/clean_raw.pkl'
    assert r.sha(source) == old['source_sha256'] and old['splits'] == r.SPLITS
    protocol = dict(setting=SETTING, source_sha=old['source_sha256'], channels=r.COLS, seeds=r.SEEDS,
                    splits=r.SPLITS, training='50% native and 50% 5-min thinning per trial per epoch',
                    selection='Robust: average native and 5-min source validation participant MAE',
                    external='native Omius, archived 160 endpoints; equal seed and fold mean',
                    trainer_sha=r.sha(r.__file__), wrapper_sha=r.sha(__file__))
    path = OUT/'protocol.json'
    if path.exists():
        assert r.read(path) == protocol
    else:
        r.save(path, protocol)
    data = r.load_source(lambda: pd.read_pickle(source))
    for split in r.SPLITS:
        fold = split['fold']
        record = r.read(CACHE/'cache_records'/f'fold{fold}.json')
        assert record['protocol_sha'] == r.sha(CACHE/'protocol.json')
        assert record['split'] == split and record['views'] == [0,2,3,5]
        for key in ['raw_path', 'metadata']:
            name = record[key]
            assert r.sha(name) == record['hashes'][name]
        cache = r.CachedInputs(record)
        assert cache.raw.shape[-2:] == (2,4)
        for seed in r.SEEDS:
            r.train_fit(data, split, SETTING, seed, cache)
    for split in r.SPLITS:
        for seed in r.SEEDS:
            r.checked_record(OUT/'records'/(r.fit_tag(SETTING,split['fold'],seed)+'.json'))
    r.save(OUT/'training_complete.json',dict(fits=10, protocol_sha=r.sha(path)))


if __name__ == '__main__':
    main()
