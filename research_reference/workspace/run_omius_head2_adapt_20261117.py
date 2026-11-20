"""Head2 cadence-adaptation refits, reusing the reviewed Regional5 training loop."""
from pathlib import Path
import argparse, json, os, gc, hashlib
import numpy as np
import pandas as pd
import torch
import run_regional_neural_transfer_v2 as r
from channel_bank400_v1 import ExpandedTCN, trial_features
from regional_neural_transfer_v2_model import HeldAgeMultikernelBackbone

ROOT=Path(__file__).resolve().parent
r.OUT=ROOT/'outputs/omius_head2_adapt_20261117'
r.MOD=r.OUT/'models'
r.COLS=['head','HR']
r.SETTINGS=['MSCA_HeldAgeRateMix','HE_Sparse5','HE_RateMix']
SOURCE=ROOT/'outputs/strict_confirmation_v1/clean_raw.pkl'
EXPECTED='0e6a48308ac45f4fa510a7e9c254a943648488811bf739abbd6619af1edd5495'

def build(setting):
    if setting.startswith('HE_'):return ExpandedTCN(2,'Direct400')
    if setting=='MSCA_HeldAgeRateMix':return HeldAgeMultikernelBackbone()
    raise ValueError(setting)
r.build_setting=build

def cache_fold(d,s):
    f=s['fold'];record=r.OUT/'cache_records'/f'fold{f}.json';dest=r.MOD/'cache'/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
    if record.exists():
        rec=r.read(record);assert rec['protocol_sha']==r.sha(r.OUT/'protocol.json')
        for p,h in rec['hashes'].items():assert r.sha(p)==h
        return rec
    x,y,v,rows,ids,groups,w,scale,_=r.encoding(d,s,r.COLS);n,t=rows.shape
    assert v.sum()==17153 and len(scale['mean'])==2 and x.shape[-1]==8
    rawpath=dest/'raw.npy';hepath=dest/'he.npy';meta=dest/'metadata.npz'
    raw=np.lib.format.open_memmap(str(rawpath)+'.tmp',mode='w+',dtype='float32',shape=(4,n,t,2,4))
    he=np.lib.format.open_memmap(str(hepath)+'.tmp',mode='w+',dtype='float32',shape=(4,n,t,2,400))
    raw[:]=0;he[:]=0
    for vi,interval in enumerate(r.VIEWS):
        for i,rr in enumerate(rows):
            rr=rr[rr>=0];a=r.encode_trial(r.sparsify(d.iloc[rr],interval),scale)
            raw[vi,i,:len(rr)]=a;he[vi,i,:len(rr)]=trial_features(a)
        r.log('CACHE VIEW COMPLETE',f,interval)
    raw.flush();he.flush();del raw,he
    os.replace(str(rawpath)+'.tmp',rawpath);os.replace(str(hepath)+'.tmp',hepath)
    np.savez_compressed(meta,y=y,valid=v,rows=rows,ids=ids,weights=w)
    rec=dict(status='passed',fold=f,split=s,scale=scale,groups={k:np.asarray(z).tolist() for k,z in groups.items()},raw_path=str(rawpath),he_path=str(hepath),metadata=str(meta),views=list(r.VIEWS),shape=[4,n,t],protocol_sha=r.sha(r.OUT/'protocol.json'),hashes={str(p):r.sha(p) for p in (rawpath,hepath,meta)})
    r.save(record,rec);return rec
r.cache_fold=cache_fold

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=range(5));ap.add_argument('--setting',choices=r.SETTINGS);ap.add_argument('--seed',type=int,choices=r.SEEDS);args=ap.parse_args()
    r.runtime('cuda')
    assert r.sha(SOURCE)==EXPECTED
    d=r.load_source(lambda:pd.read_pickle(SOURCE))
    assert d.Corerectal.notna().sum()==17153
    protocol=dict(version='Head2 source-only cadence adaptation',channels=r.COLS,settings=r.SETTINGS,seeds=r.SEEDS,splits=r.SPLITS,source_sha256=EXPECTED,trainer_sha256=r.sha(ROOT/'run_regional_neural_transfer_v2.py'),wrapper_sha256=r.sha(__file__),model_sha256=r.sha(ROOT/'regional_neural_transfer_v2_model.py'),objective='Participant-weighted MAE; AdamW .001; 150 epochs max, patience20; robust validation mean native/sparse5; two seeds',cadence='50% native; sparse5 interval 5; RateMix interval 2/3/5',source_only=True)
    pp=r.OUT/'protocol.json'
    if pp.exists():assert r.read(pp)==protocol
    else:r.save(pp,protocol)
    s=r.SPLITS[0];_,_,_,rows,_,_,_,scale,_=r.encoding(d,s,r.COLS)
    g=next(g for _,g in d.groupby('trial',sort=False));raw=r.encode_trial(g.iloc[:25],scale)
    for setting in r.SETTINGS:
        kind=r.input_kind(setting);a=trial_features(raw)[None] if kind=='HE' else r.input_from_raw(raw[None],kind)
        model=build(setting).cuda();x=torch.tensor(a,device='cuda');model.train();y=model(x);assert y.shape==(1,len(raw)) and torch.isfinite(y).all();y.sum().backward();assert all(z.grad is None or torch.isfinite(z.grad).all() for z in model.parameters());del model,x,y
    torch.cuda.empty_cache();r.save(r.OUT/'preflight.json',dict(status='passed',source_sha=EXPECTED,channels=r.COLS,labels=17153,check='2-channel HE and 22-slot held-age GPU forward/backward',source_only=True))
    folds=range(5) if args.fold is None else [args.fold];settings=r.SETTINGS if args.setting is None else [args.setting];seeds=r.SEEDS if args.seed is None else [args.seed]
    for f in folds:
        rec=cache_fold(d,r.SPLITS[f]);cache=r.CachedInputs(rec)
        for setting in settings:
            for seed in seeds:r.train_fit(d,r.SPLITS[f],setting,seed,cache)
        del cache;gc.collect()
    all_records=[r.OUT/'records'/f'{r.fit_tag(k,f,seed)}.json' for f in range(5) for k in r.SETTINGS for seed in r.SEEDS]
    if all(p.exists() for p in all_records):
        for p in all_records:r.checked_record(p)
        r.save(r.OUT/'training_complete.json',dict(status='complete',fits=30,source_labels=17153,protocol_sha=r.sha(pp),external_datasets_loaded=False))
        r.log('ALL 30 HEAD2 FITS COMPLETE')
if __name__=='__main__':main()
