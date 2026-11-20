import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
CATS=[]
COLS=["chest","upper arm","thigh","calf","HR"]
def features(group,channels):
    fields={}
    t=group.minute
    for c in channels:
        x=group[c].astype(float)
        mask=x.notna()
        last=x.ffill()
        recorded=t.where(mask)
        fields[c+'_last']=last
        fields[c+'_observed']=mask.astype(float)
        fields[c+'_age']=t-recorded.ffill()
        for n in [5,15,30,60]:
            r=x.rolling(n,min_periods=1)
            fields[f'{c}_mean{n}']=r.mean()
            fields[f'{c}_n{n}']=r.count()
            if n<=15:
                count=r.count()
                st=recorded.rolling(n,min_periods=1).sum()
                stt=(recorded*t).rolling(n,min_periods=1).sum()
                stx=(t*x).rolling(n,min_periods=1).sum()
                denom=stt-st**2/count
                fields[f'{c}_slope{n}']=((stx-st*r.sum()/count)/denom).where((count>=2)&(denom>1e-10))
            else:
                fields[f'{c}_ema{n}']=last.ewm(span=n,adjust=False,min_periods=1).mean()
        for n in (5,15,30):
            lag=last.shift(n)
            fields[f'{c}_lag{n}']=lag
            fields[f'{c}_delta{n}']=last-lag
        r=x.rolling(15,min_periods=1)
        fields[c+'_range15']=r.max()-r.min()
        fields[c+'_fast_slow']=last.ewm(span=5,adjust=False).mean()-last.ewm(span=30,adjust=False).mean()
    return pd.DataFrame(fields,index=group.index)

def buller(heart,trial,initial=36.8):
    output=np.empty(len(heart))
    for i,hr in enumerate(heart):
        if i==0 or trial[i]!=trial[i-1]:
            core,variance=initial,0.
        variance+=.022**2
        if np.isfinite(hr):
            deriv=384.4286-2*4.5714*core
            gain=variance*deriv/(variance*deriv**2+18.88**2)
            estimated_hr=-7887.1+384.4286*core-4.5714*core**2
            core+=gain*(hr-estimated_hr)
            variance*=1-gain*deriv
        output[i]=core
    return output

class PlainCNN(nn.Module):
    """Five undilated causal layers, receptive field = 141 original minutes."""
    def __init__(self, features):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(features if i == 0 else 24, 24, 29) for i in range(5))
        self.norms = nn.ModuleList(nn.LayerNorm(24) for _ in self.layers)
        self.drop = nn.Dropout(.1)
        self.out = nn.Linear(24, 1)

    def forward(self, x):
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            z = conv(F.pad(x.transpose(1, 2), (28, 0))).transpose(1, 2)
            z = self.drop(F.silu(norm(z)))
            x = z if i == 0 else x + z
        return self.out(x).squeeze(-1)

class Recurrent(nn.Module):
    def __init__(self, kind, features):
        super().__init__()
        cell = {"LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.net = cell(features, 64, num_layers=2, dropout=.1, batch_first=True)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        return self.out(self.net(x)[0]).squeeze(-1)

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

def sparsify(g,interval):
    if interval==0:return g.copy()
    assert interval in (2,3,5)
    q=g.copy();minutes=g.minute.to_numpy(float)
    for c in COLS:
        keep=np.zeros(len(g),bool);previous=-np.inf
        for i in np.flatnonzero(g[c].notna().to_numpy()):
            if minutes[i]-previous>=interval:
                keep[i]=True;previous=minutes[i]
        q.loc[~keep,c]=np.nan
    return q

def encode_trial(g,scale):
    z=g[COLS].astype(float);fresh=z.notna();held=z.ffill()
    age=np.column_stack([(g.minute-g.minute.where(fresh[c]).ffill()).fillna(0).to_numpy()/60 for c in COLS])
    values=((held-np.asarray(scale['mean']))/np.asarray(scale['std'])).fillna(0).to_numpy()
    return np.stack([values,fresh.to_numpy(float),age,held.notna().to_numpy(float)],-1).astype(np.float32)
