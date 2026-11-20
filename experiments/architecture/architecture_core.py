import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
tf = F
CATS = []
WINDOWS = (3,5,7,10,12,15,20,25,30,45,60,90)
STATS = ("mean","std","min","q25","median","q75","max","slope_per_min","last_minus_first","last_minus_mean","coverage","observed_span_fraction","observations_2plus")
MODELS = ("Shared400","Direct400")
class Block(nn.Module):
 def __init__(self,width,dilation):
  super().__init__();self.d=dilation
  self.depth=nn.ModuleList([nn.Conv1d(width,width,3,dilation=dilation,groups=width) for _ in range(2)])
  self.mix=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(2)])
  self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
  self.drop=nn.Dropout(.1)
 def forward(self,x):
  z=x
  for conv,mix,norm in zip(self.depth,self.mix,self.norm):
   z=self.step(z,conv,mix,norm)
  return x+z
 def step(self,z,conv,mix,norm):
  z=mix(conv(tf.pad(z,(2*self.d,0))));return self.drop(tf.silu(norm(z.transpose(1,2)).transpose(1,2)))

class TCN(nn.Module):
 def __init__(self,features,width):
  super().__init__();self.embed=nn.Conv1d(features,width,1)
  self.blocks=nn.Sequential(*[Block(width,2**i) for i in range(8)])
  self.out=nn.Conv1d(width,1,1)
 def forward(self,x):return self.out(self.blocks(tf.silu(self.embed(x.transpose(1,2))))).squeeze(1)

def encoding(d,s,cols):
 tr=d.Participant.isin(s['train']);out={};spec={}
 for c in cols:
  x=d[c]
  if c in CATS:
   levels=sorted(x[tr].dropna().astype(str).unique());spec[c]={'levels':levels}
   for level in levels:out[c+'='+level]=pd.Series(np.where(x.notna(),(x.astype(str)==level).astype(float),np.nan),index=d.index)
  else:out[c]=pd.to_numeric(x,errors='coerce')
 z=pd.DataFrame(out);mu=z[tr].mean().fillna(0);sd=z[tr].std().fillna(1).clip(lower=.1)
 spec.update({'columns':list(z),'mean':mu.tolist(),'std':sd.tolist()})
 groups=list(d.groupby('trial',sort=False));T=max(len(g) for _,g in groups);assert T<=1021
 X=np.zeros((len(groups),T,4*len(z.columns)),np.float32);Y=np.zeros((len(groups),T),np.float32);V=np.zeros_like(Y,bool);rows=np.full(Y.shape,-1,int);ids=[]
 ym=float(d.loc[tr,'Corerectal'].mean());ys=float(d.loc[tr,'Corerectal'].std());spec.update(ymean=ym,ystd=ys)
 def transform(g):
  r=z.loc[g.index];last=r.ffill();obs=r.notna();age=np.column_stack([(g.minute-g.minute.where(obs[c]).ffill()).fillna(0).to_numpy()/60 for c in z])
  return np.stack([((last-mu)/sd).fillna(0).to_numpy(),obs.to_numpy(float),age,last.notna().to_numpy(float)],-1).reshape(len(g),-1).astype(np.float32)
 for k,(_,g) in enumerate(groups):
  assert np.allclose(np.diff(g.minute),1);n=len(g);X[k,:n]=transform(g)
  for cut in [1,min(11,n),min(65,n)]:assert np.allclose(X[k,:cut],transform(g.iloc[:cut]))
  yy=g.Corerectal.to_numpy();Y[k,:n]=np.nan_to_num((yy-ym)/ys);V[k,:n]=np.isfinite(yy);rows[k,:n]=g.index;ids.append(g.Participant.iloc[0])
 ids=np.array(ids);ix={role:np.flatnonzero(np.isin(ids,s[role])) for role in ['train','validation','test']}
 W=np.zeros_like(Y)
 for pid in s['train']:
  mask=(ids[:,None]==pid)&V;W[mask]=1/(len(s['train'])*mask.sum())
 return X,Y,V,rows,ids,ix,W,spec,z

def metric(d,rows,p):
 rr=rows.flatten();pp=p.flatten();ok=rr>=0;rr=rr[ok];pp=pp[ok];ok=d.Corerectal.iloc[rr].notna().to_numpy();rr=rr[ok];pp=pp[ok]
 q=d.iloc[rr][['source_excel_row','Participant','Condition','trial','minute','phase_5','Corerectal']].copy();q['prediction']=pp
 err=(q.prediction-q.Corerectal).abs().groupby(q.Participant).mean().mean();return float(err),q

def feature_names():
    names = ['last_z', 'observed', 'age_hours', 'ever_observed']
    names += [f'window{w}_{s}' for w in WINDOWS for s in STATS]
    names += [f'lag{k}_{s}' for k in range(1, 61) for s in ('observed_z', 'observed')]
    names += [f'past_block{b}_{s}' for b in range(1, 31) for s in ('mean', 'std', 'slope_per_min', 'coverage')]
    assert len(set(names)) == len(names) == 400
    return names

def trial_features(encoded):
    """One real trial, [minutes, channels, 4] from the frozen training-fold encoder."""
    n, channels, slots = encoded.shape
    assert slots == 4 and n > 0
    result = np.zeros((n, channels, 400), dtype=np.float32)
    result[..., :4] = encoded
    observed = encoded[..., 1].astype(bool)
    frame = pd.DataFrame(np.where(observed, encoded[..., 0], np.nan), dtype=float)
    times = np.broadcast_to(np.arange(n)[:, None], (n, channels))
    stamps = pd.DataFrame(np.where(observed, times, np.nan))
    column_ids = np.arange(channels)[None, :]
    measured = frame.to_numpy()

    def summaries(width, detailed):
        roll = frame.rolling(width, min_periods=1)
        t_roll = stamps.rolling(width, min_periods=1)
        count = roll.count().to_numpy()
        mean = roll.mean().to_numpy()
        std = roll.std(ddof=0).to_numpy()
        sum_t = t_roll.sum().to_numpy()
        sum_tt = stamps.pow(2).rolling(width, min_periods=1).sum().to_numpy()
        sum_tx = (stamps * frame).rolling(width, min_periods=1).sum().to_numpy()
        denominator = sum_tt - sum_t**2 / np.maximum(1, count)
        slope = np.divide(sum_tx - sum_t * mean, denominator, out=np.zeros_like(mean),
                          where=(count >= 2) & (denominator > 0))
        coverage = count / width  # before trial start contributes no observations
        if not detailed:
            return mean, std, slope, coverage
        first_t = t_roll.min().to_numpy()
        last_t = t_roll.max().to_numpy()
        first = measured[np.nan_to_num(first_t).astype(int), column_ids]
        last = measured[np.nan_to_num(last_t).astype(int), column_ids]
        return (mean, std, roll.min(), roll.quantile(.25), roll.median(), roll.quantile(.75),
                roll.max(), slope, np.where(count > 0, last-first, 0),
                np.where(count > 0, last-mean, 0), coverage,
                (last_t-first_t)/(width-1), count >= 2)

    offset = 4
    for width in WINDOWS:
        for values in summaries(width, detailed=True):
            result[..., offset] = np.nan_to_num(np.asarray(values), nan=0.)
            offset += 1
    for lag in range(1, 61):
        if lag < n:
            result[lag:, :, offset] = np.where(observed[:-lag], encoded[:-lag, :, 0], 0)
            result[lag:, :, offset+1] = observed[:-lag]
        offset += 2
    blocks = np.nan_to_num(np.stack(summaries(4, detailed=False), axis=-1), nan=0.)
    for block in range(30):
        lag = 4*block + 1  # block 1 = t-4..t-1, block 30 = t-120..t-117
        if lag < n:
            result[lag:, :, offset:offset+4] = blocks[:-lag]
        offset += 4
    assert offset == 400 and np.isfinite(result).all()
    return result

class ExpandedTCN(nn.Module):
    """Original 64-wide TCN; the added feature path starts at zero."""
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in MODELS:
            raise ValueError(mode)
        self.mode = mode
        self.backbone = TCN(channels*4, 64)
        with torch.random.fork_rng(devices=[]):
            self.extra = nn.Linear(396, 4, bias=False) if mode == 'Shared400' else nn.Linear(channels*396, 64, bias=False)
            nn.init.zeros_(self.extra.weight)

    def forward(self, x):
        base, extra = x[..., :4], x[..., 4:]
        if self.mode == 'Shared400':
            return self.backbone((base + self.extra(extra)).flatten(2))
        initial = self.backbone.embed(base.flatten(2).transpose(1, 2)) + self.extra(extra.flatten(2)).transpose(1, 2)
        return self.backbone.out(self.backbone.blocks(F.silu(initial))).squeeze(1)
