"""Source-only regional-panel training and frozen Hotwater transfer.

External labels never enter fitting, selection, scaling or ensemble weights.
This retrains input-adapted versions; it is not a test of unchanged All22 weights.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import gc,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor,Pool
from prepare_multikernel_backbone_v1 import ROOT,SPLITS,SEEDS,CHANNELS,load_data,save_json,sha
from run_raw_tcn_all_v1 import TCN,encoding
from channel_bank400_v1 import ExpandedTCN,trial_features
from multikernel_backbone_model_v1 import build_model
from external_transfer_v1_data import PANELS,load_hotwater,manifest
OUT=ROOT/'outputs/external_transfer_v1'
MOD=Path('REPLACE_WITH_MODEL_DIR/external_transfer_v1')
WEIGHT=ROOT/'external/TabPFN35/tabpfn-v3.5-20260909.safetensors'
METHODS=['TabPFN','ExtraTrees','CatBoost','TCN','MSCA','HE_TCN','BlendFixed','BlendVal']
KEYS=['Participant','trial','minute','Corerectal','phase_5','source_excel_row']
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
torch.set_num_threads(4)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def runtime():
 import psutil
 from tabpfn_memory_runtime_v1 import configure_pfn_chunks
 configure_pfn_chunks()
 from threadpoolctl import threadpool_limits
 psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:4]);threadpool_limits(4)
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.65)
 for p in [OUT,OUT/'predictions',OUT/'records',MOD]:p.mkdir(parents=True,exist_ok=True)
def data():
 d=load_data().copy();d['dataset']='PROSPIE';d.trial='prospie_'+d.trial.astype(str)
 return pd.concat([d,load_hotwater()],ignore_index=True)
def protocol():
 files=['run_external_transfer_v1.py','report_external_transfer_v1.py','external_transfer_v1_data.py','tabpfn_memory_runtime_v1.py','run_raw_tcn_all_v1.py','channel_bank400_v1.py','multikernel_backbone_model_v1.py','outputs/raw_tcn_all_v1/raw.pkl','outputs/cpu_baselines_v1/splits.json']
 return dict(hashes={f:sha(ROOT/f) for f in files},external=manifest(),panels=PANELS,splits=SPLITS,seeds=SEEDS,
 task='Source-only panel retraining, followed by zero external adaptation. Regional sensor match, not exact sensor/device equivalence. Current rectal temperature estimation.',
 fitting='Only24 PROSPIE train people;8 validation select neural checkpoint, PFNmean/median and blend weights;8 test people source OOF. External9 people never used to select anything. Each source fold predicts all external people; final external prediction is fixed mean of5 source models, NOT external fivefold CV.',
 neural='TCN64, MK357(MSCA)22slots fixed chosen panel, Direct400(HE_TCN) selected panel. Two seeds per panel/fold. AdamWlr.001wd.001,batch8wholetrials,150epoch/patience20,clip1,personMAE. Train-only normalization; no phase/activity/Tc history input. Dedicated panel models, not unchanged shared seven-panel checkpoints.',
 tables='23causal features/channel. CatBoost1500 depth6 lr.035 l2=5 earlystop100 source personMAE. ExtraTrees400 minleaf2 maxfeatures1.0 train-median imputer, personweights. PFN3.5 frozenweights2estimators full train context FP32CUDA cachehost query16 sourcevalidationmean/median. No sampleweights in PFN.',
 blends='Four branches PFN,ExtraTrees,MSCA2seedmean,HE_TCN2seedmean; fixed25% and sourcevalidation simplex weights in eighths. No external selection. TCN/Cat are matched-input baselines.',
 sampling='External1minute clock; only original5minute values observed. Never interpolate labels;180actual Tr scores. Source native acquisition unchanged. Missing HR not repeated as new observation. Passive heating has no P1-P5 assignments.',
 physiological='Buller published HR-EKF reference fixed36.8 initial,P0; only real HR updates, predict-only missingminutes. Not our one-channel main model. No exact ESWA/Richmond comparator: required head/neck measurements absent.',
 limits='9 external people18trials; adjacent times not independent. Person-bootstrapCI5000, unadjusted multiplicity. Source splits repeatedly explored; external protocol frozen before scoring. Device/site, passive heating and sampling-rate distribution shift all combined.',expected_models=90)
def features(g,cols):
 f={};t=g.minute
 for c in cols:
  x=g[c].astype(float);mask=x.notna();last=x.ffill();ts=t.where(mask)
  f[c+'_last']=last;f[c+'_obs']=mask.astype(float);f[c+'_age']=t-ts.ffill()
  for n in [5,15,30,60]:
   r=x.rolling(n,min_periods=1);count=r.count();f[f'{c}_mean{n}']=r.mean();f[f'{c}_n{n}']=count
   if n<=15:
    st=ts.rolling(n,min_periods=1).sum();den=(ts*t).rolling(n,min_periods=1).sum()-st**2/count
    f[f'{c}_slope{n}']=(((t*x).rolling(n,min_periods=1).sum()-st*r.sum()/count)/den).where((count>=2)&(den>1e-10))
   else:f[f'{c}_ema{n}']=last.ewm(span=n,adjust=False).mean()
  for n in [5,15,30]:f[f'{c}_lag{n}']=last.shift(n);f[f'{c}_delta{n}']=last-last.shift(n)
  r=x.rolling(15,min_periods=1);f[c+'_range']=r.max()-r.min();f[c+'_fastslow']=last.ewm(span=5,adjust=False).mean()-last.ewm(span=30,adjust=False).mean()
 return pd.DataFrame(f,index=g.index).replace([np.inf,-np.inf],np.nan)
def weights(d,i):
 p=d.Participant.iloc[i];w=1/p.map(p.value_counts()).to_numpy();return w/w.mean()
def mae(d,i,p):return float(pd.Series(abs(p-d.Corerectal.iloc[i].to_numpy()),index=d.Participant.iloc[i]).groupby(level=0).mean().mean())
def indices(d,s):
 out={k:np.flatnonzero(d.Participant.isin(s[r])&d.Corerectal.notna()) for k,r in [('train','train'),('validation','validation'),('source_test','test')]}
 out['external']=np.flatnonzero(d.dataset.eq('Hotwater')&d.Corerectal.notna());return out
def write_pred(d,i,p,tag,role):
 assert len(i)==len(p) and np.isfinite(p).all()
 q=d.iloc[i][KEYS].copy();q['prediction']=p;path=OUT/'predictions'/f'{tag}_{role}.csv.gz';q.to_csv(path,index=False);return str(path)
def done(tag):
 p=OUT/'records'/f'{tag}.json'
 if not p.exists():return False
 r=read(p)
 for f,h in r['hashes'].items():assert sha(f)==h,(tag,f)
 return True
def commit(tag,paths,**info):save_json(OUT/'records'/f'{tag}.json',dict(status='passed',hashes={str(p):sha(p) for p in paths},**info))
def tables(d,s,panel,cols):
 x=pd.concat([features(g,cols) for _,g in d.groupby('trial',sort=False)]).sort_index();ii=indices(d,s);tr,va=ii['train'],ii['validation']
 for kind in ['ExtraTrees','CatBoost','TabPFN']:
  tag=f'{panel}_fold{s["fold"]}_{kind}'
  if done(tag):continue
  print('TABLE START',tag,flush=True);output=None
  if kind=='ExtraTrees':
   m=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('model',ExtraTreesRegressor(n_estimators=400,min_samples_leaf=2,n_jobs=4,random_state=20261030))]);m.fit(x.iloc[tr],d.Corerectal.iloc[tr],model__sample_weight=weights(d,tr));path=MOD/(tag+'.joblib');joblib.dump(m,path,compress=3)
  elif kind=='CatBoost':
   m=CatBoostRegressor(iterations=1500,depth=6,learning_rate=.035,l2_leaf_reg=5,loss_function='RMSE',eval_metric='MAE',thread_count=4,random_seed=20261030,allow_writing_files=False,verbose=False)
   m.fit(Pool(x.iloc[tr],d.Corerectal.iloc[tr],weight=weights(d,tr)),eval_set=Pool(x.iloc[va],d.Corerectal.iloc[va],weight=weights(d,va)),early_stopping_rounds=100);path=MOD/(tag+'.cbm');m.save_model(str(path))
  else:
   from tabpfn import TabPFNRegressor
   from tabpfn.model_loading import save_fitted_tabpfn_model,load_fitted_tabpfn_model
   m=TabPFNRegressor(model_path=str(WEIGHT),device='cuda',n_estimators=2,n_preprocessing_jobs=4,random_state=20261030,fit_mode='fit_with_cache',inference_precision=torch.float32,memory_saving_mode=True,keep_cache_on_device=False,show_progress_bar=False);m.fit(x.iloc[tr],d.Corerectal.iloc[tr].to_numpy())
  def predict(i):
   return np.concatenate([m.predict(x.iloc[i[k:k+16]],output_type=output) for k in range(0,len(i),16)]) if kind=='TabPFN' else m.predict(x.iloc[i])
  if kind=='TabPFN':
   values={}
   for output in ['mean','median']:values[output]=mae(d,va,predict(va))
   output=min(values,key=values.get);path=MOD/(tag+'.tabpfn_fit');save_fitted_tabpfn_model(m,path)
  preds={role:predict(i) for role,i in ii.items() if role!='train'};del m;gc.collect();torch.cuda.empty_cache()
  if kind=='ExtraTrees':m=joblib.load(path)
  elif kind=='CatBoost':m=CatBoostRegressor(thread_count=4);m.load_model(str(path))
  else:
   m=load_fitted_tabpfn_model(path,device='cpu');m.keep_cache_on_device=False;m.executor_.keep_cache_on_device=False;m.memory_saving_mode=True;m.to('cuda')
  pick=ii['external'][:64];replay=float(abs(predict(pick)-preds['external'][:64]).max());assert replay<5e-5
  files=[path]+[write_pred(d,ii[r],p,tag,r) for r,p in preds.items()]
  commit(tag,files,model=str(path),split=s,output=output,replay_error=replay,features=list(x));del m;gc.collect();torch.cuda.empty_cache();print('TABLE DONE',tag,flush=True)
def arrays(d,s,cols):
 x,y,v,rows,ids,groups,w,scale,_=encoding(d,s,cols);slots=x.reshape(*x.shape[:2],len(cols),4)
 groups['source_test']=groups.pop('test');groups['external']=np.flatnonzero(ids>=1000)
 bank=np.zeros((*x.shape[:2],len(cols),400),np.float32);msca=np.zeros((*x.shape[:2],22,3),np.float32)
 for j,c in enumerate(cols):
  k=CHANNELS.index(c);msca[...,k,0]=np.where(slots[...,j,1]!=0,slots[...,j,0],0);msca[...,k,1]=slots[...,j,1];msca[...,k,2]=1
 for i,r in enumerate(rows):
  n=int((r>=0).sum());bank[i,:n]=trial_features(slots[i,:n])
 return {k:torch.tensor(a,device='cuda') for k,a in [('TCN',x),('MSCA',msca),('HE_TCN',bank),('y',y),('w',w)]},rows,groups,scale
def model(kind,c):return TCN(c*4,64) if kind=='TCN' else build_model('MK357') if kind=='MSCA' else ExpandedTCN(c,'Direct400')
@torch.inference_mode()
def npredict(m,x,ix,rows,scale):
 m.eval();pp=[];rr=[]
 for start in range(0,len(ix),8):
  b=ix[start:start+8];pp.extend((m(x[b]).cpu().numpy()*scale['ystd']+scale['ymean']).ravel());rr.extend(rows[b].ravel())
 out=pd.Series(pp,index=rr);return out[out.index>=0].sort_index()
def neural(d,s,panel,cols):
 a,rows,groups,scale=arrays(d,s,cols);ii=indices(d,s)
 for kind in ['TCN','MSCA','HE_TCN']:
  tag=f'{panel}_fold{s["fold"]}_{kind}'
  if done(tag):continue
  for seed in SEEDS:
   st=tag+f'_s{seed}'
   if done(st):continue
   print('NET START',st,flush=True);torch.manual_seed(seed+s['fold']);rng=np.random.default_rng(seed+s['fold']);m=model(kind,len(cols)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);best=float('inf');history=[];tr=groups['train']
   for epoch in range(1,151):
    m.train();order=rng.permutation(tr)
    for start in range(0,len(order),8):
     b=order[start:start+8];opt.zero_grad(set_to_none=True);loss=((m(a[kind][b])-a['y'][b]).abs()*a['w'][b]).sum()*len(tr)/len(b);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    p=npredict(m,a[kind],groups['validation'],rows,scale);val=mae(d,ii['validation'],p.loc[ii['validation']].to_numpy());history.append(val)
    if val<best:best=val;bestepoch=epoch;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    if epoch==1 or epoch%10==0:print(st,'epoch',epoch,'val',val,flush=True)
    if epoch-bestepoch>=20:break
   m.load_state_dict(state);path=MOD/(st+'.pt');torch.save(dict(state=state,split=s,scale=scale,channels=cols,kind=kind),path)
   preds={r:npredict(m,a[kind],groups[r],rows,scale).loc[ii[r]].to_numpy() for r in ['validation','source_test','external']}
   m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);replay=float(abs(npredict(m,a[kind],groups['external'],rows,scale).loc[ii['external']].to_numpy()-preds['external']).max());assert replay<5e-5
   cache=OUT/'records'/(st+'.npz');np.savez_compressed(cache,**preds);commit(st,[path,cache],model=str(path),history=history,bestepoch=bestepoch,split=s,replay_error=replay);del m,opt;gc.collect();torch.cuda.empty_cache()
  files=[]
  for r in ['validation','source_test','external']:
   p=np.mean([np.load(OUT/'records'/f'{tag}_s{seed}.npz')[r] for seed in SEEDS],0);files.append(write_pred(d,ii[r],p,tag,r))
  commit(tag,files,seeds=SEEDS)
 del a;gc.collect();torch.cuda.empty_cache()
def blend(d,s,panel):
 ii=indices(d,s);parts={r:np.column_stack([pd.read_csv(OUT/'predictions'/f'{panel}_fold{s["fold"]}_{k}_{r}.csv.gz').prediction for k in ['TabPFN','ExtraTrees','MSCA','HE_TCN']]) for r in ['validation','source_test','external']}
 candidates=[np.array([a,b,c,8-a-b-c])/8 for a in range(9) for b in range(9-a) for c in range(9-a-b)]
 selected=min(candidates,key=lambda w:mae(d,ii['validation'],parts['validation']@w))
 for name,w in [('BlendFixed',np.ones(4)/4),('BlendVal',selected)]:
  tag=f'{panel}_fold{s["fold"]}_{name}'
  if done(tag):continue
  paths=[write_pred(d,ii[r],p@w,tag,r) for r,p in parts.items()];commit(tag,paths,weights=w.tolist(),selection='PROSPIE validation people only')
def buller(d):
 p=[];old=None
 for r in d.itertuples():
  if r.trial!=old:core,var,old=36.8,0.,r.trial
  var+=.022**2
  if np.isfinite(r.HR):
   deriv=384.4286-2*4.5714*core;gain=var*deriv/(var*deriv**2+18.88**2);core+=gain*(r.HR-(-7887.1+384.4286*core-4.5714*core**2));var*=1-gain*deriv
  p.append(core)
 return np.array(p)
def prepare():
 d=data();s=SPLITS[0];x,y,v,rows,ids,groups,w,scale,_=encoding(d,s,PANELS['Regional5']);assert v[:155].sum()==17162 and v[155:].sum()==180
 g=d.iloc[:100];np.testing.assert_allclose(features(g,PANELS['Regional5']).iloc[:17],features(g.iloc[:17],PANELS['Regional5']),equal_nan=True)
 torch.manual_seed(20261105);m=TCN(20,64).eval();xx=torch.tensor(x[:2,:30]);p=m(xx);zz=xx.clone();zz[:,15:]+=20
 assert torch.allclose(p[:,:15],m(zz)[:,:15],atol=1e-6) and torch.allclose(p[:,:15],m(xx[:,:15]),atol=1e-5)
 mutated=d.copy();mutated.loc[mutated.dataset.eq('Hotwater'),'Corerectal']=90
 sp=encoding(mutated,s,PANELS['Regional5'])[7];assert scale==sp
 pr=protocol();path=OUT/'protocol.json'
 if path.exists():assert read(path)==pr
 else:save_json(path,pr)
 save_json(OUT/'preflight.json',dict(status='passed',source_rows=17264,source_labels=17162,external_rows=828,external_labels=180,causal_features=True,causal_network=True,external_labels_do_not_affect_scaling=True))
 print('EXTERNAL PREFLIGHT PASSED',flush=True)
def train():
 assert read(OUT/'protocol.json')==protocol();assert read(OUT/'preflight.json')['status']=='passed'
 import msvcrt
 with (OUT/'training.lock').open('a+b') as lock:
  if not lock.tell():lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  d=data()
  for s in SPLITS:
   for panel,cols in PANELS.items():tables(d,s,panel,cols);neural(d,s,panel,cols);blend(d,s,panel)
  ex=d[d.dataset.eq('Hotwater')];q=ex[KEYS].copy();q['prediction']=buller(ex);q=q[q.Corerectal.notna()];q.to_csv(OUT/'predictions/Buller_external.csv.gz',index=False)
  save_json(OUT/'completion.json',dict(status='complete',models=90,panels=list(PANELS),external_people=9,external_labels=180));print('EXTERNAL TRAINING COMPLETE',flush=True)
if __name__=='__main__':
 runtime();prepare() if sys.argv[1]=='prepare' else train()
