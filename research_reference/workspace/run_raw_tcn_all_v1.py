"""Causal raw-sequence TCN and matched sensor-set CatBoost exploration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='12'
from pathlib import Path
import json,hashlib,datetime,time,copy,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as tf
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/raw_tcn_all_v1';OUT.mkdir(exist_ok=True)
MOD=Path('REPLACE_WITH_MODEL_DIR/raw_tcn_all_v1');MOD.mkdir(parents=True,exist_ok=True)
SRC=Path('REPLACE_WITH_DATA_DIR/prospie/Prospie dataset_open access V3.xlsx')
BASE=['abdomen','calf','chest','foot','hand','head','lower arm','lower back','thigh','upper arm','upper back','HR']
PHYS=BASE+['Ins skin','Tamb','BR']
SENSOR=PHYS+['Temp','Humidity','Wind','Tfront','Rhfront','Tback','Rhback','Solar present','Overall (1.00-1.50m)','Middle (1.00-1.50m)','Overall (1.00-1.70m)']
ALL=SENSOR+['Sex','Age','Bodymasskg','Heightcm','Bodyfat%','VO2peakmlkgmin','Acclimationstatus','Trainingstatus','Clothing','workrest','Speedkm.hr-1','Gradient%','TS','TC','TP','SW','RPE']
SETS={'Base12':BASE,'NeckResp15':PHYS,'Sensors26':SENSOR,'Context43':ALL}
CATS=['Sex','Acclimationstatus','Trainingstatus','Clothing','workrest','Solar present']
SPLITS=json.loads((ROOT/'outputs/cpu_baselines_v1/splits.json').read_text())
SEEDS=[20261103,20261104]
torch.set_num_threads(12);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def prepare():
 import openpyxl
 w=openpyxl.load_workbook(SRC,read_only=True,data_only=True);it=w['Data'].iter_rows(max_col=58)
 a=next(it);b=next(it);names=[str(y.value or x.value or '').strip() for x,y in zip(a,b)]
 records=[];redcounts={c:0 for c in names}
 for rn,row in enumerate(it,3):
  if row[3].value is None:continue
  rec={'source_excel_row':rn}
  for name,c in zip(names,row):
   red=bool(c.font and c.font.color and c.font.color.type=='rgb' and c.font.color.rgb[-6:]=='FF0000')
   redcounts[name]+=int(red)
   rec[name]=np.nan if red or c.value==9999 else c.value
  records.append(rec)
 w.close();d=pd.DataFrame(records)
 raw=pd.read_csv(ROOT/'outputs/phase_distill_v2/raw_model_signals.csv')
 keys=raw[['source_excel_row','Participant','Condition','trial','minute']]
 d=keys.merge(d.drop(columns=['Participant','Condition']),on='source_excel_row',validate='one_to_one')
 labels=pd.read_csv(ROOT/'outputs/five_phase_v1/labels/minute_labels.csv')[['source_excel_row','phase_5']]
 d=d.merge(labels,on='source_excel_row',validate='one_to_one')
 meta=pd.read_csv(ROOT/'outputs/cpu_baselines_v1/metadata.csv')
 assert len(d)==17264 and d.Corerectal.notna().sum()==17162 and d.trial.nunique()==155
 assert np.allclose(d.loc[d.Corerectal.notna(),'Corerectal'],meta.Corerectal)
 for c in BASE:assert np.allclose(pd.to_numeric(d[c]),raw[c],equal_nan=True),c
 for c in ALL:
  if c not in CATS:d[c]=pd.to_numeric(d[c],errors='coerce')
 d.to_pickle(OUT/'raw.pkl')
 audit=[]
 for c in ALL:
  ok=d[c].notna();gaps=d.loc[ok].groupby('trial').minute.diff().dropna()
  audit.append({'field':c,'usable_rows':int(ok.sum()),'coverage':float(ok.mean()),'people':int(d.loc[ok,'Participant'].nunique()),'trials':int(d.loc[ok,'trial'].nunique()),'red_cells_masked':redcounts[c],'recorded_interval_counts':gaps.value_counts().to_dict()})
 dump(OUT/'input_audit.json',{'fields':audit,'sets':SETS,'rows':len(d),'labels':17162,'source_sha':sha(SRC),'excluded':{'Corerectal':'target only','Δcore':'derived from target','SweatProductionL':'whole-trial post minus pre mass; future leakage','Activity code':'includes stopping reasons and core threshold','Participant/Condition/Time':'identifiers or protocol clock not model inputs','ml.kg.min':'5-minute collection timing unresolved','HRV six columns':'300-beat/5-minute processing window timing unresolved','HRtype':'measurement-method metadata; not physiological measurement'}})
 pd.DataFrame([{k:v for k,v in z.items() if k!='recorded_interval_counts'} for z in audit]).to_csv(OUT/'input_coverage.csv',index=False)
 print('PREPARED',len(d),len(ALL),flush=True)

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

def checks(d):
 s=SPLITS[0];X,Y,V,rows,ids,ix,W,sp,z=encoding(d,s,ALL)
 net=TCN(X.shape[-1],32).cuda().eval();x=torch.tensor(X[:2],device='cuda')
 with torch.no_grad():
  p=net(x);xx=x.clone();xx[:,65:]=torch.randn_like(xx[:,65:]);q=net(xx)
  assert torch.allclose(p[:,:65],q[:,:65],atol=1e-6)
  assert torch.allclose(p[:,:65],net(x[:,:65]),atol=1e-5)
 net.train();net(x).mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
 dump(OUT/'pipeline_checks.json',{'status':'passed','future_perturbation':'passed','prefix_features':'passed','prefix_network':'passed','gradient':'passed','max_length':X.shape[1],'receptive_field':1021,'parameters32':sum(p.numel() for p in net.parameters()),'peak_GPU_MB':torch.cuda.max_memory_allocated()/1e6})
 del net,x;torch.cuda.empty_cache()

def protocol():
 return {'script_sha':sha(__file__),'report_sha':sha(ROOT/'report_raw_tcn_all_v1.py'),'raw_sha':sha(OUT/'raw.pkl'),'splits':SPLITS,'sets':SETS,'seeds':SEEDS,'task':'current core temperature, not future forecasting','device':'CUDA float32 TCN; CatBoost CPU12','TCN':{'widths':[32,64],'dilations':[2**i for i in range(8)],'kernel':3,'convs_per_block':2,'receptive_field':1021,'epochs':150,'patience':20,'lr':.001,'weight_decay':.001,'dropout':.1,'batch_trials':8,'loss':'person-balanced MAE','selection':'validation person MAE only'},'CatBoost':{'depth':6,'iterations':1500,'lr':.035,'l2':5,'patience':100,'loss':'RMSE','selection':'validation person MAE','features':'existing444 unchanged; extra channels18 causal last/mask/age + mean/count/slope5/15/30/60 + delta5/15/30; NaN=-999'},'rules':['No true core history, future input, experiment ID or stage input.','Every original timepoint retained; absent labels masked only in loss/scoring.','Normalization and categorical vocabulary fitted only on training people; no per-trial centering.','All sequences causal and reset by trial.','Context43 includes subjective ratings/activity/individual context; not wearable-only.','Repeated exploratory splits; no confirmatory claim; no test-based parameter revision.','Two TCN seeds reported separately and equal average as preregistered; never choose best test seed.']}

def train(d):
 pr=protocol();assert read(OUT/'protocol.json')==pr
 for s in SPLITS:
  for name,cols in SETS.items():
   X,Y,V,rows,ids,ix,W,sp,z=encoding(d,s,cols)
   for width in [32,64]:
    for seed in SEEDS:
     tag=f'fold{s["fold"]}_{name}_TCN{width}_seed{seed}';done=OUT/(tag+'.json')
     if done.exists():continue
     t=time.perf_counter();torch.manual_seed(seed+s['fold']);np.random.seed(seed+s['fold']);net=TCN(X.shape[-1],width).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
     tx,ty,tw=[torch.tensor(a,device='cuda') for a in [X,Y,W]]
     @torch.no_grad()
     def pred(role):
      net.eval();return np.concatenate([net(tx[b]).cpu().numpy()*sp['ystd']+sp['ymean'] for b in np.array_split(ix[role],max(1,int(np.ceil(len(ix[role])/8))))])
     best=None;history=[]
     print('START',tag,'parameters',sum(p.numel() for p in net.parameters()),flush=True)
     for epoch in range(150):
      net.train();order=np.random.permutation(ix['train'])
      for b in [order[i:i+8] for i in range(0,len(order),8)]:
       p=net(tx[b]);loss=(abs(p-ty[b])*tw[b]).sum()*len(order)/len(b);assert torch.isfinite(loss)
       opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
      err,_=metric(d,rows[ix['validation']],pred('validation'));history.append(err)
      if best is None or err<best[0]:best=(err,copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()}),epoch)
      if epoch%10==0:print(tag,'epoch',epoch+1,'val',round(err,5),flush=True)
      if epoch-best[2]>=20:break
     net.load_state_dict(best[1]);path=MOD/(tag+'.pt');torch.save({'state':best[1],'preprocessing':sp,'width':width,'features':X.shape[-1],'split':s},path)
     for role in ['validation','test']:
      _,q=metric(d,rows[ix[role]],pred(role));q.to_csv(OUT/(tag+'_'+role+'.csv'),index=False)
     # Reload and replay entire held-out sequences, not isolated history-free rows.
     saved=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(saved['state']);_,q=metric(d,rows[ix['test']],pred('test'));ref=pd.read_csv(OUT/(tag+'_test.csv'));replay=float(abs(q.prediction.to_numpy()-ref.prediction).max());assert replay<5e-5
     dump(done,{'status':'complete','model':str(path),'sha':sha(path),'validation_MAE':best[0],'best_epoch':best[2]+1,'history':history,'seconds':time.perf_counter()-t,'parameters':sum(p.numel() for p in net.parameters()),'replay_error':replay})
     del tx,ty,tw,net,opt;torch.cuda.empty_cache();print('DONE',tag,flush=True)
   cat(d,s,name,cols,z)
 dump(OUT/'completion.json',{'status':'complete','TCN_fits':80,'CatBoost_fits':20});print('TRAINING COMPLETE',flush=True)

def cat(d,s,name,cols,z):
 from catboost import CatBoostRegressor,Pool
 tag=f'fold{s["fold"]}_{name}_CatBoost';done=OUT/(tag+'.json')
 if done.exists():return
 # Existing 444 features unchanged; add identical causal summaries for new numeric/one-hot channels.
 manifest=read(ROOT/'outputs/temporal_capacity_v1/feature_manifest.json');old=pd.read_pickle(ROOT/'outputs/temporal_capacity_v1/features502.pkl')[manifest['feature_sets']['F444']]
 parts=[]
 extra=[c for c in z if c not in BASE]
 for _,g in d.groupby('trial',sort=False):
  feats={}
  for c in extra:
   x=z.loc[g.index,c];obs=x.notna();last=x.ffill();tt=g.minute
   feats[c+'_last']=last;feats[c+'_observed']=obs.astype(float);feats[c+'_age']=tt-tt.where(obs).ffill()
   for k in [5,15,30,60]:
    r=x.rolling(k,min_periods=1);n=r.count();sx=r.sum();st=tt.where(obs).rolling(k,min_periods=1).sum();st2=(tt*tt).where(obs).rolling(k,min_periods=1).sum();stx=(tt*x).rolling(k,min_periods=1).sum();den=st2-st*st/n
    feats[f'{c}_mean{k}']=r.mean();feats[f'{c}_n{k}']=n;feats[f'{c}_slope{k}']=((stx-st*sx/n)/den).where((n>=2)&(den>1e-10))
   for k in [5,15,30]:feats[f'{c}_delta{k}']=last-last.shift(k)
  parts.append(pd.DataFrame(feats,index=g.index))
 f=pd.concat([old,pd.concat(parts).sort_index()],axis=1).replace([np.inf,-np.inf],np.nan).fillna(-999.)
 ii={role:np.flatnonzero(d.Participant.isin(s[role])&d.Corerectal.notna()) for role in ['train','validation','test']}
 def weights(i):return 1/d.Participant.iloc[i].map(d.Participant.iloc[i].value_counts()).to_numpy()
 print('START',tag,'features',f.shape[1],flush=True)
 model=CatBoostRegressor(iterations=1500,depth=6,learning_rate=.035,l2_leaf_reg=5,loss_function='RMSE',eval_metric='MAE',thread_count=12,random_seed=SEEDS[0],verbose=False,allow_writing_files=False)
 tr,va=ii['train'],ii['validation'];wt=weights(tr);wv=weights(va)
 model.fit(f.iloc[tr],d.Corerectal.iloc[tr],sample_weight=wt/wt.mean(),eval_set=Pool(f.iloc[va],d.Corerectal.iloc[va],weight=wv/wv.mean()),early_stopping_rounds=100)
 path=MOD/(tag+'.cbm');model.save_model(str(path))
 for role in ['validation','test']:
  i=ii[role];q=d.iloc[i][['source_excel_row','Participant','Condition','trial','minute','phase_5','Corerectal']].copy();q['prediction']=model.predict(f.iloc[i]);q.to_csv(OUT/(tag+'_'+role+'.csv'),index=False)
 reload=CatBoostRegressor();reload.load_model(str(path));replay=float(np.max(abs(reload.predict(f.iloc[ii['test']])-q.prediction)));assert replay<1e-8
 dump(done,{'status':'complete','model':str(path),'sha':sha(path),'trees':model.tree_count_,'features':list(f),'replay_error':replay});print('DONE',tag,flush=True)

if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'train'
 if mode=='prepare':
  prepare();d=pd.read_pickle(OUT/'raw.pkl');checks(d);p=protocol()
  if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==p
  else:dump(OUT/'protocol.json',p)
 elif mode=='train':train(pd.read_pickle(OUT/'raw.pkl'))
