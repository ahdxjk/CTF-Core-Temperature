"""Independent ESWA 2026 implementations; assumptions documented in README.

Not author code. EKF is a specified reconstruction because full equations and
coefficients are absent from the target paper. No import starts training.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn

METHODS=['PhysicsOnly','EKF_Reconstructed','CNNOnly','LSTMOnly','GRUOnly',
         'HybridCNNFull','HybridCNNSimple','HybridLSTM','HybridGRU']
NEURAL=METHODS[2:]
@dataclass
class PhysicsConfig:
 core_heat_capacity:float=3650.
 blood_heat_capacity:float=4180.
 blood_density:float=1050.
 conductance:float=5.28
 core_set:float=36.8
 skin_set:float=33.7
 basal_flow:float=6.3
 dilation:float=200.
 constriction:float=.1
 area:float=1.8
 mass:float=70.
 age:float=30.
 skin_mass_ratio:float=.042
 resting_hr:float=60.
 basal_metabolic:float=60.
 initial_core:float=36.8
 # Explicit sensitivity option, not chosen by test accuracy.
 metabolic_mode:str='area_consistent'
def metabolic_rate(hr,c):
 maximum=(19.45-.133*c.age)*c.mass
 fraction=(np.asarray(hr)-c.resting_hr)/(208-.7*c.age-c.resting_hr)
 if c.metabolic_mode=='area_consistent':return c.basal_metabolic+(maximum/c.area-c.basal_metabolic)*fraction
 if c.metabolic_mode=='literal_divide':return (c.basal_metabolic+(maximum-c.basal_metabolic)*fraction)/c.area
 raise ValueError(c.metabolic_mode)
def physical_step(core,hr,skin,ambient,rh_percent,seconds,c=None,simplified=False):
 c=c or PhysicsConfig();m=float(metabolic_rate(hr,c))
 flow=c.basal_flow+c.dilation*max(0.,core-c.core_set)/(1+c.constriction*max(0.,c.skin_set-skin))
 qskin=(c.blood_heat_capacity*c.blood_density*flow/(3600*1000)+c.conductance)*(core-skin)
 ta=25. if simplified else ambient
 pa=1584. if simplified else rh_percent/100*.61078*np.exp(17.27*ta/(ta+237.3))*1000
 qresp=.0014*m*(34-ta)+.0023*m*(44-.0075*pa)
 return core+c.area/((1-c.skin_mass_ratio)*c.mass*c.core_heat_capacity)*(m-qskin-qresp)*seconds
def physical_trace(inputs,minutes,trial,c=None,simplified=False):
 """inputs columns HR,head,Temp,Humidity; input contains no target."""
 c=c or PhysicsConfig();p=np.empty(len(inputs),float)
 for i,(hr,skin,ta,rh) in enumerate(inputs):
  if i==0 or trial[i]!=trial[i-1]:p[i]=c.initial_core
  else:
   dt=60*(minutes[i]-minutes[i-1]);assert dt>0
   p[i]=physical_step(p[i-1],hr,skin,ta,rh,dt,c,simplified)
 if not np.isfinite(p).all():raise FloatingPointError('Nonfinite physical trajectory; do not silently clip or drop rows')
 return p

class EKF:
 """Scalar affine state in Tc-37; quadratic HR/head observation; train-only LS.

This is NOT a claimed exact reconstruction of the authors' unspecified EKF.
"""
 def fit(self,core,observations,minutes,trial,train_mask):
  y=np.asarray(core);z=y-37.;mask=np.asarray(train_mask,bool)
  pair=mask[1:]&mask[:-1]&np.isfinite(y[1:])&np.isfinite(y[:-1])&(trial[1:]==trial[:-1])&np.isclose(np.diff(minutes),1)
  if pair.sum()<10:raise ValueError('Insufficient adjacent training pairs')
  a=np.column_stack([np.ones(pair.sum()),z[:-1][pair]]);self.transition=np.linalg.lstsq(a,z[1:][pair],rcond=None)[0]
  self.Q=max(float(np.var(z[1:][pair]-a@self.transition)),1e-6)
  ok=mask&np.isfinite(y)&np.isfinite(observations).all(1)
  if ok.sum()<10:raise ValueError('Insufficient observed training pairs')
  design=np.column_stack([np.ones(ok.sum()),z[ok],z[ok]**2]);self.observation=np.linalg.lstsq(design,observations[ok],rcond=None)[0]
  errors=observations[ok]-design@self.observation;self.R=np.cov(errors,rowvar=False)+np.eye(2)*1e-6
  self.initial=36.8;self.initial_variance=.1**2;return self
 def predict(self,observations,minutes,trial):
  out=np.empty(len(trial));state=self.initial-37.;variance=self.initial_variance
  for i,obs in enumerate(observations):
   new=i==0 or trial[i]!=trial[i-1]
   if new:state=self.initial-37.;variance=self.initial_variance
   else:
    if not np.isclose(minutes[i]-minutes[i-1],1):raise ValueError('EKF transition calibrated for 1 minute only')
    state=self.transition[0]+self.transition[1]*state;variance=self.transition[1]**2*variance+self.Q
   use=np.flatnonzero(np.isfinite(obs))
   if len(use):
    coeff=self.observation[:,use];expected=np.array([1,state,state**2])@coeff
    h=coeff[1]+2*state*coeff[2];r=self.R[np.ix_(use,use)]
    innovation_cov=variance*np.outer(h,h)+r;k=np.linalg.solve(innovation_cov,variance*h)
    state=state+k@(obs[use]-expected);variance=(1-k@h)**2*variance+k@r@k
   out[i]=state+37.
  if not np.isfinite(out).all():raise FloatingPointError('Nonfinite EKF output')
  return out
 def state(self):return {k:np.asarray(v).tolist() for k,v in vars(self).items()}
 @classmethod
 def restore(cls,state):
  m=cls()
  for k,v in state.items():setattr(m,k,np.asarray(v) if isinstance(v,list) else v)
  return m

class WindowNetwork(nn.Module):
 def __init__(self,architecture,features):
  super().__init__();self.architecture=architecture
  if architecture=='CNN':
   layers=[];prev=features
   for i,width in enumerate([32,64,96]):
    layers += [nn.Conv1d(prev,width,3,padding=1),nn.BatchNorm1d(width),nn.ReLU()]
    if i<2:layers += [nn.MaxPool1d(2,ceil_mode=True)]
    prev=width
   self.conv=nn.Sequential(*layers);self.head=nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))
  else:
   unit=nn.LSTM if architecture=='LSTM' else nn.GRU
   first,second=(64,32) if architecture=='LSTM' else (80,40)
   self.rnn1=unit(features,first,batch_first=True);self.rnn2=unit(first,second,batch_first=True)
   self.dropout=nn.Dropout(.2);self.head=nn.Linear(second,1)
 def forward(self,x):
  if self.architecture=='CNN':return self.head(self.conv(x.transpose(1,2)).mean(-1)).squeeze(-1)
  a,_=self.rnn1(x);b,_=self.rnn2(self.dropout(a));return self.head(self.dropout(b[:,-1])).squeeze(-1)
def architecture(method):return 'CNN' if 'CNN' in method else 'LSTM' if 'LSTM' in method else 'GRU'
def inputs_for(method,sensors,physics_full,physics_simple):
 if method=='HybridCNNSimple':return np.column_stack([sensors[:,:2],physics_simple]),physics_simple
 if method.startswith('Hybrid'):return np.column_stack([sensors,physics_full]),physics_full
 return sensors.copy(),np.zeros(len(sensors))
def causal_fill(sensors,trial,train_mask,medians=None):
 import pandas as pd
 medians=np.nanmedian(sensors[train_mask],axis=0) if medians is None else np.asarray(medians)
 if not np.isfinite(medians).all():raise ValueError('All-missing training sensor')
 filled=pd.DataFrame(sensors).groupby(np.asarray(trial),sort=False).ffill().fillna(dict(enumerate(medians))).to_numpy()
 return filled,medians
def window_indices(trial,k):
 result=np.empty((len(trial),k),int);start=0
 for i in range(len(trial)):
  if i==0 or trial[i]!=trial[i-1]:start=i
  result[i]=np.maximum(np.arange(i-k+1,i+1),start)
 return result
def minmax_fit(x):
 lo=np.min(x,axis=0);span=np.max(x,axis=0)-lo;return lo,np.where(span>1e-8,span,1.)
def dtw(y,p):
 previous=np.full(len(p)+1,np.inf);previous[0]=0.
 for a in y:
  current=np.full(len(p)+1,np.inf)
  for j,b in enumerate(p,1):current[j]=(a-b)**2+min(previous[j],current[j-1],previous[j-1])
  previous=current
 return float(np.sqrt(previous[-1]))
def metrics(frame):
 from sklearn.metrics import roc_auc_score
 y=frame.Corerectal.to_numpy();p=frame.prediction.to_numpy();e=p-y;den=((y-y.mean())**2).sum()
 positive=y>=38.;pred=p>=38.
 return {'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(abs(e))),'MD':float(e.mean()),'SD':float(e.std(ddof=0)),
         'R2':float(1-(e**2).sum()/den) if den>0 else None,
         'DTW_trial_mean':float(np.mean([dtw(g.Corerectal.to_numpy(),g.prediction.to_numpy()) for _,g in frame.groupby('trial',sort=False)])),
         'AUC38':float(roc_auc_score(positive,p)) if len(np.unique(positive))==2 else None,
         'Sensitivity38':float(pred[positive].mean()) if positive.any() else None,
         'Specificity38':float((~pred[~positive]).mean()) if (~positive).any() else None,'Accuracy38':float((pred==positive).mean())}
