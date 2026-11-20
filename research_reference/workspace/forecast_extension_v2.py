"""Fourteen-sensor factual forecasts; independently frozen extension."""
import os,json,hashlib
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from run_raw_tcn_all_v1 import TCN,BASE,encoding
from prepare_multikernel_backbone_v1 import ROOT,SPLITS,SEEDS,load_data,save_json,sha
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
torch.set_num_threads(4)
OUT=ROOT/'outputs/forecast_extension_v2'
MODEL=Path('REPLACE_WITH_MODEL_DIR/forecast_extension_v2')
CHANNELS=BASE+['Ins skin','Tamb']
HORIZONS=[0,1,5,10,15,30]
METHODS=['TCNNoPlan','TCNPlan','CatNoPlan','CatPlan','TabPFNNoPlan','TabPFNPlan']
WEIGHT=ROOT/'external/TabPFN35/tabpfn-v3.5-20260909.safetensors'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def runtime():
    import psutil
    from threadpoolctl import threadpool_limits
    from tabpfn_memory_runtime_v1 import configure_pfn_chunks
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:4]);threadpool_limits(limits=4)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.65)
    configure_pfn_chunks(cells=65536)

def labels(d):
    y=np.full((len(d),len(HORIZONS)),np.nan);target=np.full_like(y,-1,dtype=int)
    actions=np.zeros((len(d),30),np.float32);available=np.zeros_like(actions,bool)
    for g in d.groupby('trial',sort=False).indices.values():
        assert np.all(np.diff(d.minute.iloc[g])==1)
        y[g,0]=d.Corerectal.iloc[g];target[g,0]=g
        for h in range(1,31):
            src=g[:-h];dst=g[h:]
            actions[src,h-1]=d.workrest.iloc[dst].eq(1);available[src,h-1]=d.workrest.iloc[dst].isin([1,2])
            if h in HORIZONS:
                j=HORIZONS.index(h);y[src,j]=d.Corerectal.iloc[dst];target[src,j]=dst
    valid=np.isfinite(y)
    for j,h in enumerate(HORIZONS):
        if h:valid[:,j]&=available[:,:h].all(1)
    y[~valid]=np.nan
    return y,target,actions,valid

def table_features(g):
    fields={};t=g.minute
    for c in CHANNELS:
        x=pd.to_numeric(g[c],errors='coerce');mask=x.notna();last=x.ffill();recorded=t.where(mask)
        fields[c+'_last']=last;fields[c+'_observed']=mask.astype(float);fields[c+'_age']=t-recorded.ffill()
        for n in [5,15,30,60]:
            r=x.rolling(n,min_periods=1);fields[f'{c}_mean{n}']=r.mean();fields[f'{c}_n{n}']=r.count()
            if n<=15:
                count=r.count();st=recorded.rolling(n,min_periods=1).sum();stt=(recorded*t).rolling(n,min_periods=1).sum();stx=(t*x).rolling(n,min_periods=1).sum();denom=stt-st**2/count
                fields[f'{c}_slope{n}']=((stx-st*r.sum()/count)/denom).where((count>=2)&(denom>1e-10))
            else:fields[f'{c}_ema{n}']=last.ewm(span=n,adjust=False,min_periods=1).mean()
        for n in (5,15,30):fields[f'{c}_lag{n}']=last.shift(n);fields[f'{c}_delta{n}']=last-last.shift(n)
        r=x.rolling(15,min_periods=1);fields[c+'_range15']=r.max()-r.min();fields[c+'_fast_slow']=last.ewm(span=5,adjust=False).mean()-last.ewm(span=30,adjust=False).mean()
    q=pd.DataFrame(fields,index=g.index);q['current_work']=g.workrest.eq(1).astype(float);q['current_activity_known']=g.workrest.isin([1,2]).astype(float)
    assert q.shape[1]==324
    return q

def pack(d,s,device='cuda'):
    x,_,_,rows,ids,groups,_,scale,_=encoding(d,s,CHANNELS)
    y,target,actions,valid=labels(d);safe=rows.clip(min=0);present=rows>=0
    work=d.workrest.eq(1).to_numpy()[safe];known=d.workrest.isin([1,2]).to_numpy()[safe]
    x=np.concatenate([x,work[...,None],known[...,None]],-1).astype(np.float32);x[~present]=0
    yy=np.nan_to_num((y[safe]-scale['ymean'])/scale['ystd']).astype(np.float32);v=valid[safe]&present[...,None]
    a=actions[safe];a[~present]=0;w=np.zeros_like(yy)
    for j,h in enumerate(HORIZONS):
        for pid in s['train']:
            mask=(ids[:,None]==pid)&v[...,j];assert mask.any();w[...,j][mask]=1/(len(s['train'])*len(HORIZONS)*mask.sum())
    assert np.isclose(w.sum(),1)
    return [torch.tensor(z,device=device) for z in (x,yy,a,w)],rows,groups,scale

class ForecastTCN(nn.Module):
    def __init__(self,planned):
        super().__init__();self.planned=planned
        self.encoder=TCN(58,64);self.encoder.out=nn.Identity()
        self.decoder=nn.Sequential(nn.Linear(64+61,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x,actions):
        hidden=self.encoder(x).transpose(1,2);out=[]
        for horizon in HORIZONS:
            mask=(torch.arange(30,device=x.device)<horizon).to(x.dtype)
            if not self.planned:mask=mask*0
            allowed=actions*mask
            features=torch.cat([hidden,allowed,mask.expand_as(allowed),torch.full_like(actions[...,:1],horizon/30)],-1)
            out.append(self.decoder(features).squeeze(-1))
        return torch.stack(out,-1)

def person_mae(d,indices,p,y):
    return float(pd.Series(abs(p-y),index=d.Participant.iloc[indices]).groupby(level=0).mean().mean())

def write_predictions(d,s,method,seed,p):
    y,target,actions,valid=labels(d);parts=[]
    for j,h in enumerate(HORIZONS):
        ix=np.flatnonzero(d.Participant.isin(s['test'])&valid[:,j]);dst=target[ix,j]
        q=d.iloc[ix][['source_excel_row','Participant','Condition','trial','minute','phase_5']].copy()
        q['horizon']=h;q['target_excel_row']=d.source_excel_row.iloc[dst].to_numpy();q['truth']=y[ix,j];q['prediction']=p[ix,j]
        q['method']=method;q['seed']=str(seed);q['fold']=s['fold'];q['current_activity']=np.where(d.workrest.iloc[ix].eq(1),'work',np.where(d.workrest.iloc[ix].eq(2),'rest','unknown'))
        aa=actions[ix,:h];q['future_path']='current' if not h else np.where(aa.sum(1)==h,'work',np.where(aa.sum(1)==0,'rest','mixed'))
        assert np.isfinite(q.prediction).all();parts.append(q)
    return pd.concat(parts,ignore_index=True)

def contract():
    files=['forecast_extension_v2.py','prepare_forecast_extension_v2.py','run_forecast_extension_v2.py','report_forecast_extension_v2.py','tabpfn_memory_runtime_v1.py','run_raw_tcn_all_v1.py','prepare_multikernel_backbone_v1.py','outputs/raw_tcn_all_v1/raw.pkl','outputs/cpu_baselines_v1/splits.json']
    return dict(hashes={f:sha(ROOT/f) for f in files},splits=SPLITS,seeds=SEEDS,channels=CHANNELS,horizons=HORIZONS,
        neural_fits=20,cat_fits=55,tabpfn_fits=55,
        task='Current Tc plus factual forecasts at1/5/10/15/30minutes. Strictly no true Tc history/current Tc input. No future physiological measurements. Same origins per horizon across methods; current Tc need not be labelled to forecast.',
        inputs='14 physiological sensors; current work/rest and known mask. Neural58 current dimensions from causal last/mask/age/ever plus2 activity flags. Tabular23features/sensor=322+2 activity flags. Future h binary work/rest entries only in Plan. Plan beyond h has zero influence.',
        neural='Shared causalTCN64 (8doubledepthwise blocks dilations1..128), same decoder64+30futureactions+30prefixmask+1horizon->64->1. NoPlan/Plan equal architecture, two seeds; six outputs include h0. Raw58 input. Entire trial chronology retained.',
        neural_training='FP32 CUDA,CPU4affinity4,TF32off,allocator65%. AdamWlr.001wd.001batch8fulltrials,clip1,max150patience20; loss mean of per-person per-horizon MAE over h0/1/5/10/15/30. Validation same six-horizon/person mean; fixed seed average. No fallback selection.',
        cat='CPU4 CatBoostdepth6,1000trees,lr.035,l2=5,RMSEtraining/personweights; validation personweightedMAE earlystop80. Independent model/horizon. h0 shared between Plan/NoPlan. 5fold*(6NoPlan+5Plan)=55. Seed20261030.',
        tabpfn='Official frozen3.5 weights,2estimators,fullcontext,FP32CUDA,memorysaving,hostKVcache,query16,CPU4,independent-row chunk cells65536; no dropped context rows or weight finetuning. Same324historical/current features as Cat plush futureplan bits for Plan. Source8person validation selects mean/median pointestimate. h0 shared. 55fits. No sample weights in PFN context, unlike personweighted Cat/TCN.',
        tabpfn_weight_sha=sha(WEIGHT),
        references='Persistence and clippedlineartrend from TCNNoPlan, CatNoPlan and TabPFNNoPlan own h0 prediction; not measuredTc. Linear slope(currentestimate minus own estimate5minago)/elapsed,clip +/-.1C/min; insufficient history uses available prefix. Existing12route earlier study only historical reference, not matched-input ablation.',
        evaluation='Per-person MAE/RMSE/MD, five origin phases, current/future activity path, all-horizon-common-origin sensitivity, two seeds and fixedmean, paired10000personbootstrap. Forced all-work/all-rest alternatives are unvalidated and not scored as counterfactual ground truth.',
        limits='Fixed splits repeatedly explored. Not independent confirmation. Unadjusted multiplicity. Supplied future plan is retrospective factual protocol, not automatically known future or proven causal effect. Cat vs neural use same rawchannels but different representations/objectives.')

def assert_frozen():assert read(OUT/'protocol.json')==contract()
