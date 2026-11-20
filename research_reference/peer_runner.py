"""Portable strict-data peer reruns. No original outputs/checkpoints are modified.

preflight: audit inputs, folds, causality and implementations, no fitting.
smoke: tiny CPU/GPU numerical passes; never exported as scientific results.
run: frozen original grids selected using strict source validation only.
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'2')
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['TABPFN_NO_BROWSER']='1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='true'
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import frozen_helpers as h
import eswa_baseline_models as eswa

SEEDS=[20261103,20261104]
TABSEED=20261030
CLASSIC=['CatBoost','LightGBM','XGBoost','RBF_SVR']
SENSORS=['HR','head','Temp','Humidity']
KEYS=['source_excel_row','Participant','Condition','trial','minute','phase_5','Corerectal']
REGIONAL=['chest','upper arm','thigh','calf','HR']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');temp.replace(p)
def tsave(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');torch.save(obj,temp);temp.replace(p)
def log(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def weights(d,i):
    p=d.Participant.iloc[i];w=1/p.map(p.value_counts()).to_numpy();return w/w.mean()
def mae(d,i,p):return float(pd.Series(abs(p-d.Corerectal.iloc[i].to_numpy()),index=d.Participant.iloc[i]).groupby(level=0).mean().mean())
def rows(d,s):return {r:np.flatnonzero(d.Participant.isin(s[r])&d.Corerectal.notna()) for r in ('train','validation','test')}
def prediction(d,i,p,path,**labels):
    assert len(i)==len(p) and np.isfinite(p).all()
    q=d.iloc[i][[k for k in KEYS+['dataset'] if k in d]].copy();q['prediction']=np.asarray(p)
    for k,v in labels.items():q[k]=v
    path.parent.mkdir(parents=True,exist_ok=True);q.to_csv(path,index=False,compression='gzip')
    return path
def metrics(q):
    e=q.prediction-q.Corerectal
    p=pd.DataFrame(dict(Participant=q.Participant,ae=abs(e),se=e**2,bias=e)).groupby('Participant').mean()
    return dict(MAE=float(p.ae.mean()),RMSE=float(np.sqrt(p.se).mean()),Bias=float(p.bias.mean()),people=len(p),labels=len(q))
def complete(out,tag,files,**record):
    save(out/'records'/f'{tag}.json',dict(status='passed',protocol_sha=sha(out/'protocol.json'),
         hashes={str(Path(f).relative_to(out)):sha(f) for f in files},**record))
def done(out,tag):
    f=out/'records'/f'{tag}.json'
    if not f.exists():return False
    r=read(f);assert r['protocol_sha']==sha(out/'protocol.json'),f
    for p,digest in r['hashes'].items():assert sha(out/p)==digest,(tag,p)
    return True

def grids(name):
    if name=='CatBoost':return [dict(depth=d,l2_leaf_reg=l,learning_rate=lr) for d,l,lr in [(4,5,.035),(6,5,.035),(8,5,.035),(4,10,.035),(6,10,.035),(6,5,.07)]]
    if name=='LightGBM':return [dict(num_leaves=n,min_child_samples=m) for n in [7,15,31] for m in [20,80]]
    if name=='XGBoost':return [dict(max_depth=d,min_child_weight=w) for d in [3,5,7] for w in [5,20]]
    if name=='ExtraTrees':return [dict(min_samples_leaf=l,max_features=f) for l in [2,5,10] for f in [.5,1.]]
    if name=='RBF_SVR':return [dict(C=c,gamma=g) for c in [1.,10.,100.] for g in ['scale',.01]]
    raise ValueError(name)

def setup(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda':assert torch.cuda.is_available(),'CUDA unavailable; use --device cpu only for preflight/smoke'
    manifest=read(HERE/'input_manifest.json')
    for a in manifest['assets']:assert sha(HERE/'assets'/a['asset'])==a['sha256'],a['asset']
    d=pd.read_pickle(HERE/'assets/clean_raw.pkl');splits=read(HERE/'assets/splits.json')
    assert len(d)==17264 and d.Corerectal.notna().sum()==17153 and d.trial.nunique()==155
    assert d.index.equals(pd.RangeIndex(len(d))) and d.source_excel_row.is_unique
    assert sorted(p for s in splits for p in s['test'])==sorted(d.Participant.unique())
    for s in splits:
        groups=[set(s[r]) for r in ('train','validation','test')]
        assert list(map(len,groups))==[24,8,8] and not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    mask=pd.read_csv(HERE/'assets/mask_cells.csv')
    bykey=d.set_index('source_excel_row')
    assert all(pd.isna(bykey.loc[r.source_excel_row,r.field]) for r in mask.itertuples())
    protocol=dict(version='strict-peer-portable-v1',input_manifest=sha(HERE/'input_manifest.json'),
        code={p.name:sha(p) for p in [Path(__file__),HERE/'frozen_helpers.py',HERE/'eswa_baseline_models.py',HERE/'external_data.py']},
        splits=splits,source_labels=17153,seed_tables=TABSEED,seeds_external=SEEDS,
        grids={n:grids(n) for n in CLASSIC+['ExtraTrees']},windows=[5,10,15,20],
        selection='Source validation only. Table: participant-macro MAE; window networks: pooled RMSE; CNN full: native participantMAE; GRU full: mean native/sparse5 participantMAE.',
        table_weighting='inverse training person row count normalized mean1; TabPFN unweighted full context',
        windows_budget='200epochs maximum; Adam .001; minibatch32; MSE; patience30; strict train-only minmax',
        external_budget='150epochs; AdamW .001 wd.001; wholetrialbatch8; personMAE; clip1; patience20; seeds20261103/4; no target/phase/activity inputs',
        external_selection='CNN native; GRU robust=.5native+.5sparse5 matching archived GRU_Native Robust. Both GRU checkpoints retained.',
        no_test_selection=True,external_mean='all5folds and2seeds, no external fit/selection; stronger baselines retained')
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol,'Protocol changed; choose a fresh output directory'
    else:save(out/'protocol.json',protocol)
    for p in ['records','models','predictions','resume']: (out/p).mkdir(exist_ok=True)
    return d,splits,manifest,out

def preflight(args,d,splits,manifest,out):
    import external_data
    s=splits[0];g=next(g for _,g in d.groupby('trial',sort=False));checks={}
    for panel,cols in manifest['panels'].items():
        full=h.features(g,cols)
        for cut in (1,11,min(65,len(g))):
            np.testing.assert_allclose(full.iloc[:cut],h.features(g.iloc[:cut],cols),equal_nan=True,atol=1e-8)
        assert full.shape[1]==23*len(cols)
        checks[panel]=dict(features=full.shape[1],prefix=True)
    arr=h.encoding(d,s,REGIONAL);scale=arr[7]
    changed=d.copy();outside=~d.Participant.isin(s['train']);changed.loc[outside,REGIONAL]=12345.;changed.loc[outside,'Corerectal']=-100.
    assert h.encoding(changed,s,REGIONAL)[7]==scale
    assert int(arr[2].sum())==17153 and np.isclose(arr[6].sum(),1.)
    checks['neural_scale_source_only']=True
    for kind in ['CNN','GRU']:
        torch.manual_seed(61116);m=(h.PlainCNN(20) if kind=='CNN' else h.Recurrent('GRU',20)).eval()
        x=torch.randn(2,50,20)
        with torch.no_grad():
            p=m(x);xx=x.clone();xx[:,31:]+=100
            delta=max(float(abs(p[:,:31]-m(xx)[:,:31]).max()),float(abs(p[:,:31]-m(x[:,:31])).max()))
        assert delta<5e-5,(kind,delta);checks[kind]=dict(prefix_error=delta,parameters=sum(p.numel() for p in m.parameters()))
    np.testing.assert_allclose(h.buller([124.,111.,119.,145.],[0]*4,37.94),[37.94031,37.93962,37.93979,37.94525],atol=1.5e-5,rtol=0)
    hot=external_data.load_hotwater();bright=external_data.load_brighton()
    external={}
    for name,e in [('Hotwater',hot),('Brighton',bright)]:
        assert e.source_excel_row.is_unique and not set(e.Participant)&set(d.Participant)
        assert e.groupby('trial').minute.diff().dropna().eq(1).all()
        assert e.loc[~e.source_observed_time,REGIONAL+['Corerectal']].isna().all().all()
        external[name]=dict(labels=int(e.Corerectal.notna().sum()),rows=len(e),people=e.Participant.nunique(),trials=e.trial.nunique(),genuine_reference_alignment=True)
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scikit-learn','catboost','lightgbm','xgboost','tabpfn']}
    save(out/'preflight.json',dict(status='passed',source_labels=17153,checks=checks,external=external,versions=versions,device=args.device,threads=args.threads))
    log('PREFLIGHT PASSED',external)

def classic_create(name,cfg,threads):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.svm import SVR
    if name=='CatBoost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=1200,loss_function='RMSE',eval_metric='MAE',thread_count=threads,task_type='CPU',random_seed=TABSEED,allow_writing_files=False,verbose=False,**cfg)
    if name=='LightGBM':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=1200,learning_rate=.035,reg_lambda=5.,objective='regression',metric='l1',subsample=.9,subsample_freq=1,colsample_bytree=.9,n_jobs=threads,random_state=TABSEED,verbosity=-1,deterministic=True,force_col_wise=True,**cfg)
    if name=='XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=1200,learning_rate=.035,reg_lambda=5.,objective='reg:squarederror',eval_metric='mae',early_stopping_rounds=100,tree_method='hist',device='cpu',subsample=.9,colsample_bytree=.9,n_jobs=threads,random_state=TABSEED,**cfg)
    imputer=('imputer',SimpleImputer(strategy='median',keep_empty_features=True))
    if name=='ExtraTrees':return Pipeline([imputer,('model',ExtraTreesRegressor(n_estimators=400,max_depth=None,n_jobs=threads,random_state=TABSEED,**cfg))])
    return Pipeline([imputer,('scaler',StandardScaler()),('model',SVR(kernel='rbf',epsilon=.05,cache_size=1024,max_iter=-1,**cfg))])

def classic_fit(name,m,x,d,tr,va):
    from catboost import Pool
    from lightgbm import early_stopping,log_evaluation
    y=d.Corerectal.to_numpy();wt,wv=weights(d,tr),weights(d,va)
    if name=='CatBoost':m.fit(Pool(x.iloc[tr],y[tr],weight=wt),eval_set=Pool(x.iloc[va],y[va],weight=wv),early_stopping_rounds=100)
    elif name=='LightGBM':m.fit(x.iloc[tr],y[tr],sample_weight=wt,eval_set=[(x.iloc[va],y[va])],eval_sample_weight=[wv],eval_metric='l1',callbacks=[early_stopping(100,first_metric_only=True,verbose=False),log_evaluation(0)])
    elif name=='XGBoost':m.fit(x.iloc[tr],y[tr],sample_weight=wt,eval_set=[(x.iloc[va],y[va])],sample_weight_eval_set=[wv],verbose=False)
    else:m.fit(x.iloc[tr],y[tr],model__sample_weight=wt)
    if name=='RBF_SVR':assert m.named_steps['model'].fit_status_==0
    return m

def table_classic(args,d,s,x,panel,name,out):
    import joblib
    tag=f'table_{panel}_f{s["fold"]}_{name}'
    if done(out,tag):return
    ii=rows(d,s);tr,va=ii['train'],ii['validation'];candidates=[]
    for j,cfg in enumerate(grids(name)):
        ctag=tag+f'_candidate{j}';path=out/'models'/f'{ctag}.joblib'
        if not done(out,ctag):
            log('FIT',ctag);start=time.perf_counter();m=classic_fit(name,classic_create(name,cfg,args.threads),x,d,tr,va)
            error=mae(d,va,m.predict(x.iloc[va]));joblib.dump(m,path,compress=3)
            clone=joblib.load(path);replay=float(np.max(abs(m.predict(x.iloc[va])-clone.predict(x.iloc[va]))));assert replay<5e-5
            complete(out,ctag,[path],validation_MAE=error,parameters=cfg,model=str(path.relative_to(out)),seconds=time.perf_counter()-start,replay=replay)
            del m,clone;gc.collect()
        candidates.append(read(out/'records'/f'{ctag}.json'))
    selected=min(range(len(candidates)),key=lambda j:candidates[j]['validation_MAE'])
    save(out/'records'/f'{tag}_selection.json',dict(candidate=selected,validation_MAE=candidates[selected]['validation_MAE'],selection='strict validation only'))
    m=joblib.load(out/candidates[selected]['model']);files=[]
    for role in ['validation','test']:files.append(prediction(d,ii[role],m.predict(x.iloc[ii[role]]),out/'predictions'/f'{tag}_{role}.csv.gz',panel=panel,method=name,fold=s['fold'],role=role))
    complete(out,tag,files+[out/'records'/f'{tag}_selection.json'],candidate=selected,method=name,panel=panel,fold=s['fold'],validation_MAE=candidates[selected]['validation_MAE'])
    log('SELECTED',tag,selected)

def pfn_weight(args):
    candidates=[args.pfn_weight] if args.pfn_weight else [args.bundle_root/'assets/tabpfn/tabpfn-v3.5-20260909.safetensors',args.bundle_root/'models/tabpfn-v3.5-20260909.safetensors',args.bundle_root/'external/TabPFN35/tabpfn-v3.5-20260909.safetensors',args.bundle_root/'assets/tabpfn-v3.5-20260909.safetensors']
    for path in candidates:
        if path and path.is_file():
            assert sha(path)=='ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3','TabPFN foundation weights changed'
            return path.resolve()
    raise FileNotFoundError('Use --pfn-weight to select the bundled TabPFN v3.5 weights.')

def make_pfn(args,weight):
    from tabpfn import TabPFNRegressor
    # Bound internal row chunks without dropping any training context or queries.
    if not hasattr(TabPFNRegressor,'_unbounded_initializer'):
        TabPFNRegressor._unbounded_initializer=TabPFNRegressor._initialize_model_variables
        def initialize(self):
            size=TabPFNRegressor._unbounded_initializer(self)
            for internal in self.models_:
                if hasattr(internal,'inference_chunk_cells'):internal.inference_chunk_cells=65536
            return size
        TabPFNRegressor._initialize_model_variables=initialize
    return TabPFNRegressor(model_path=str(weight),device=args.device,n_estimators=2,fit_mode='fit_with_cache',random_state=TABSEED,n_preprocessing_jobs=2,show_progress_bar=False,inference_precision=torch.float32,memory_saving_mode=True,keep_cache_on_device=False)

def table_pfn(args,d,s,x,out):
    from tabpfn.model_loading import save_fitted_tabpfn_model,load_fitted_tabpfn_model
    tag=f'table_ESWA4_f{s["fold"]}_TabPFN_3_5'
    if done(out,tag):return
    ii=rows(d,s);tr,va=ii['train'],ii['validation'];weight=pfn_weight(args)
    log('FIT',tag,'all train rows',len(tr))
    model=make_pfn(args,weight)
    model.fit(x.iloc[tr],d.Corerectal.iloc[tr].to_numpy())
    def predict(ix,output):return np.concatenate([model.predict(x.iloc[ix[j:j+16]],output_type=output) for j in range(0,len(ix),16)])
    pv={k:predict(va,k) for k in ['mean','median']};errors={k:mae(d,va,p) for k,p in pv.items()};selected=min(errors,key=errors.get)
    selection=out/'records'/f'{tag}_selection.json';save(selection,dict(selected=selected,validation_errors=errors,context_rows=d.source_excel_row.iloc[tr].tolist(),weight_sha256=sha(weight)))
    path=out/'models'/f'{tag}.tabpfn_fit';save_fitted_tabpfn_model(model,path)
    files=[prediction(d,va,pv[selected],out/'predictions'/f'{tag}_validation.csv.gz',panel='ESWA4',method='TabPFN_3_5',fold=s['fold'],role='validation')]
    pt=predict(ii['test'],selected);files.append(prediction(d,ii['test'],pt,out/'predictions'/f'{tag}_test.csv.gz',panel='ESWA4',method='TabPFN_3_5',fold=s['fold'],role='test'))
    ref=predict(va[:8],selected);del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    model=load_fitted_tabpfn_model(path,device=args.device);replay=float(np.max(abs(predict(va[:8],selected)-ref)));assert replay<5e-5
    complete(out,tag,files+[selection,path],selected=selected,validation_MAE=errors[selected],replay=replay)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def tables(args,d,splits,manifest,out):
    panels=list(manifest['panels']) if args.panels=='all' else args.panels.split(',')
    for panel in panels:
        cols=manifest['panels'][panel];x=pd.concat([h.features(g,cols) for _,g in d.groupby('trial',sort=False)]).sort_index()
        assert x.shape==(17264,23*len(cols)) and not np.isinf(x.to_numpy()).any()
        for s in splits:
            for name in CLASSIC+(['ExtraTrees'] if panel=='ESWA4' else []):table_classic(args,d,s,x,panel,name,out)
            if panel=='ESWA4':table_pfn(args,d,s,x,out)

def physical_inputs(d,s):
    raw=d[SENSORS].to_numpy(float);filled,med=eswa.causal_fill(raw,d.trial.to_numpy(),d.Participant.isin(s['train']).to_numpy())
    full=eswa.physical_trace(filled,d.minute.to_numpy(),d.trial.to_numpy())
    simple=eswa.physical_trace(filled,d.minute.to_numpy(),d.trial.to_numpy(),simplified=True)
    return raw,filled,med,full,simple

def physiology(args,d,splits,out):
    for s in splits:
        raw,filled,med,full,simple=physical_inputs(d,s);ii=rows(d,s)
        for name in ['PhysicsOnly','EKF_Reconstructed','Buller_HR']:
            tag=f'phys_ESWA4_f{s["fold"]}_{name}'
            if done(out,tag):continue
            state=dict(split=s,medians=med.tolist(),physics=eswa.asdict(eswa.PhysicsConfig()))
            if name=='PhysicsOnly':p=full
            elif name=='Buller_HR':p=h.buller(raw[:,0],d.trial.to_numpy())
            else:
                model=eswa.EKF().fit(d.Corerectal.to_numpy(),raw[:,:2],d.minute.to_numpy(),d.trial.to_numpy(),d.Participant.isin(s['train']).to_numpy())
                state['ekf']=model.state();p=eswa.EKF.restore(state['ekf']).predict(raw[:,:2],d.minute.to_numpy(),d.trial.to_numpy())
            path=out/'models'/f'{tag}.json';save(path,state);files=[path]
            for role in ['validation','test']:files.append(prediction(d,ii[role],p[ii[role]],out/'predictions'/f'{tag}_{role}.csv.gz',panel='ESWA4',method=name,fold=s['fold'],role=role))
            complete(out,tag,files,method=name,fold=s['fold'])
            log('DONE',tag)

@torch.inference_mode()
def window_predict(net,xx,wi,ix,device):
    net.eval();return np.concatenate([net(torch.as_tensor(xx[wi[ix[j:j+256]]],dtype=torch.float32,device=device)).cpu().numpy() for j in range(0,len(ix),256)])

def neural_windows(args,d,splits,out):
    for s in splits:
        raw,filled,med,full,simple=physical_inputs(d,s);ii=rows(d,s);tr,va=ii['train'],ii['validation'];y=d.Corerectal.to_numpy()
        for method in eswa.NEURAL:
            tag=f'window_ESWA4_f{s["fold"]}_{method}'
            if done(out,tag):continue
            x,base=eswa.inputs_for(method,filled,full,simple);target=y-base
            lo,span=eswa.minmax_fit(x[d.Participant.isin(s['train'])]);ylo,yspan=eswa.minmax_fit(target[tr]);xx=(x-lo)/span;yy=(target-ylo)/yspan
            choices=[]
            for window in [5,10,15,20]:
                ctag=tag+f'_k{window}';path=out/'models'/f'{ctag}.pt';wi=eswa.window_indices(d.trial.to_numpy(),window)
                if not done(out,ctag):
                    torch.manual_seed(SEEDS[0]+s['fold']);rng=np.random.default_rng(SEEDS[0]+s['fold'])
                    net=eswa.WindowNetwork(eswa.architecture(method),x.shape[1]).to(args.device);opt=torch.optim.Adam(net.parameters(),lr=.001)
                    best=float('inf');bestepoch=-1;state=None;history=[];start=0;resume=out/'resume'/f'{ctag}.pt'
                    if resume.exists():
                        ck=torch.load(resume,map_location=args.device,weights_only=False);assert ck['protocol_sha']==sha(out/'protocol.json')
                        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu())
                        if args.device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
                        best,bestepoch,state,history,start=ck['best'],ck['bestepoch'],ck['state'],ck['history'],ck['epoch']+1
                    log('FIT',ctag,'from epoch',start+1)
                    for epoch in ([] if start and start-1-bestepoch>=30 else range(start,200)):
                        net.train();order=rng.permutation(tr)
                        for k in range(0,len(order),32):
                            b=order[k:k+32];tx=torch.as_tensor(xx[wi[b]],dtype=torch.float32,device=args.device);ty=torch.as_tensor(yy[b],dtype=torch.float32,device=args.device)
                            loss=((net(tx)-ty)**2).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
                        p=window_predict(net,xx,wi,va,args.device)*yspan+ylo+base[va];error=float(np.sqrt(np.mean((p-y[va])**2)));history.append(error)
                        if error<best:best,bestepoch,state=error,epoch,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        tsave(resume,dict(protocol_sha=sha(out/'protocol.json'),model=net.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],best=best,bestepoch=bestepoch,state=state,history=history,epoch=epoch))
                        if epoch%10==0:log(ctag,'epoch',epoch+1,'validationRMSE',error)
                        if epoch-bestepoch>=30:break
                    net.load_state_dict(state);ref=window_predict(net,xx,wi,va,args.device)
                    tsave(path,dict(weights=state,architecture=eswa.architecture(method),features=x.shape[1],window=window,xlo=lo,xspan=span,ylo=ylo,yspan=yspan,medians=med,physics=eswa.asdict(eswa.PhysicsConfig()),method=method,split=s))
                    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['weights']);replay=float(abs(ref-window_predict(net,xx,wi,va,args.device)).max());assert replay<5e-5
                    complete(out,ctag,[path],window=window,validation_RMSE=best,best_epoch=bestepoch+1,history=history,model=str(path.relative_to(out)),replay=replay)
                    del net,opt;gc.collect()
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                choices.append(read(out/'records'/f'{ctag}.json'))
            chosen=min(choices,key=lambda c:c['validation_RMSE']);selection=out/'records'/f'{tag}_selection.json';save(selection,dict(window=chosen['window'],validation_RMSE=chosen['validation_RMSE']))
            ck=torch.load(out/chosen['model'],map_location='cpu',weights_only=False);net=eswa.WindowNetwork(ck['architecture'],ck['features']).to(args.device);net.load_state_dict(ck['weights']);wi=eswa.window_indices(d.trial.to_numpy(),ck['window']);files=[selection]
            for role in ['validation','test']:
                i=ii[role];p=window_predict(net,xx,wi,i,args.device)*yspan+ylo+base[i];files.append(prediction(d,i,p,out/'predictions'/f'{tag}_{role}.csv.gz',panel='ESWA4',method=method,fold=s['fold'],role=role))
            complete(out,tag,files,window=chosen['window'],validation_RMSE=chosen['validation_RMSE']);del net

def full_model(kind):return h.PlainCNN(20) if kind=='CNN' else h.Recurrent('GRU',20)
@torch.inference_mode()
def sequence_predict(model,x,groups,rows,scale,device):
    model.eval();pieces=[]
    for start in range(0,len(groups),8):
        batch=groups[start:start+8];p=model(torch.as_tensor(x[batch],device=device)).cpu().numpy()*scale['ystd']+scale['ymean']
        rr=rows[batch];valid=rr>=0;pieces.append(pd.Series(p[valid],index=rr[valid]))
    return pd.concat(pieces).sort_index()

def external_train(args,d,splits,out):
    for s in splits:
        x,y,valid,rowids,pids,groups,w,scale,_=h.encoding(d,s,REGIONAL);tr=groups['train'];va=groups['validation'];ii=rows(d,s)
        sparse=pd.concat([h.sparsify(g,5) for _,g in d.groupby('trial',sort=False)]).sort_index()
        sparse_x=np.zeros_like(x)
        for j,(_,g) in enumerate(sparse.groupby('trial',sort=False)):sparse_x[j,:len(g)]=h.encode_trial(g,scale).reshape(len(g),20)
        for kind in ['CNN','GRU']:
            for seed in SEEDS:
                tag=f'external_{kind}_f{s["fold"]}_s{seed}'
                if done(out,tag):continue
                torch.manual_seed(seed+s['fold']);rng=np.random.default_rng(seed+s['fold']);model=full_model(kind).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                best={'Native':float('inf'),'Robust':float('inf')};bestepoch={'Native':0,'Robust':0};states={};history=[];start=1;stop_policy='Native' if kind=='CNN' else 'Robust';resume=out/'resume'/f'{tag}.pt'
                if resume.exists():
                    ck=torch.load(resume,map_location=args.device,weights_only=False);assert ck['protocol_sha']==sha(out/'protocol.json')
                    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu())
                    if args.device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
                    best,bestepoch,states,history,start=ck['best'],ck['bestepoch'],ck['states'],ck['history'],ck['epoch']+1
                log('FIT',tag,'stop policy',stop_policy,'from epoch',start)
                for epoch in ([] if history and history[-1]['epoch']-bestepoch[stop_policy]>=20 else range(start,151)):
                    model.train();order=rng.permutation(tr)
                    for j in range(0,len(order),8):
                        b=order[j:j+8];xx,yy,ww=[torch.as_tensor(a[b],device=args.device) for a in [x,y,w]]
                        opt.zero_grad(set_to_none=True);loss=((model(xx)-yy).abs()*ww).sum()*len(tr)/len(b);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    p=sequence_predict(model,x,va,rowids,scale,args.device).loc[ii['validation']].to_numpy();native=mae(d,ii['validation'],p)
                    # CNN never uses sparse scores in checkpoint or early-stop selection.
                    if kind=='GRU':
                        p=sequence_predict(model,sparse_x,va,rowids,scale,args.device).loc[ii['validation']].to_numpy();sparse_mae=mae(d,ii['validation'],p);robust=.5*(native+sparse_mae)
                    else:sparse_mae=None;robust=native
                    history.append(dict(epoch=epoch,native_MAE=native,sparse5_MAE=sparse_mae,robust_MAE=robust))
                    for policy,value in [('Native',native),('Robust',robust)]:
                        tolerance=1e-12 if kind=='GRU' else 0
                        if value<best[policy]-tolerance:best[policy]=value;bestepoch[policy]=epoch;states[policy]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    tsave(resume,dict(protocol_sha=sha(out/'protocol.json'),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],best=best,bestepoch=bestepoch,states=states,history=history,epoch=epoch))
                    if epoch==1 or epoch%10==0:log(tag,'epoch',epoch,'validation',best[stop_policy])
                    if epoch-bestepoch[stop_policy]>=20:break
                files=[]
                for policy in (['Native'] if kind=='CNN' else ['Native','Robust']):
                    model.load_state_dict(states[policy]);path=out/'models'/f'{tag}_{policy}.pt';tsave(path,dict(state=states[policy],channels=REGIONAL,scale=scale,split=s,seed=seed,kind=kind,policy=policy));files.append(path)
                    for role in ['validation','test']:
                        p=sequence_predict(model,x,groups[role],rowids,scale,args.device).loc[ii[role]].to_numpy();files.append(prediction(d,ii[role],p,out/'predictions'/f'{tag}_{policy}_{role}.csv.gz',panel='Regional5',method=f'{kind}_{policy}',fold=s['fold'],seed=seed,role=role))
                    ref=sequence_predict(model,x,va,rowids,scale,args.device)
                    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);assert float(abs(ref-sequence_predict(model,x,va,rowids,scale,args.device)).max())<5e-5
                complete(out,tag,files,best=best,bestepoch=bestepoch,history=history,selection_policy=stop_policy)
                del model,opt;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()

def external_evaluate(args,d,splits,out):
    import external_data
    # External tables are loaded after every source-only fit has completed.
    for kind in ['CNN','GRU']:
        for s in splits:
            for seed in SEEDS:assert done(out,f'external_{kind}_f{s["fold"]}_s{seed}')
    ext=pd.concat([external_data.load_hotwater(),external_data.load_brighton()],ignore_index=True)
    if 'phase_5' not in ext:ext['phase_5']='U'
    ext['phase_5']=ext.phase_5.fillna('U');parts=list(ext.groupby('trial',sort=False))
    for kind in ['CNN','GRU']:
        for policy in (['Native'] if kind=='CNN' else ['Native','Robust']):
            predictions=[]
            for s in splits:
                for seed in SEEDS:
                    tag=f'external_{kind}_f{s["fold"]}_s{seed}';ck=torch.load(out/'models'/f'{tag}_{policy}.pt',map_location='cpu',weights_only=False);assert ck['split']==s and ck['channels']==REGIONAL
                    model=full_model(kind).to(args.device);model.load_state_dict(ck['state']);scale=ck['scale'];x=np.zeros((len(parts),max(len(g) for _,g in parts),20),np.float32);rr=np.full(x.shape[:2],-1,int)
                    for j,(_,g) in enumerate(parts):x[j,:len(g)]=h.encode_trial(g,scale).reshape(len(g),20);rr[j,:len(g)]=g.index
                    pp=sequence_predict(model,x,np.arange(len(parts)),rr,scale,args.device);assert pp.index.equals(ext.index)
                    predictions.append(pp.to_numpy());del model
            mean=np.mean(predictions,axis=0)
            for dataset in ['Hotwater','Brighton']:
                ix=np.flatnonzero(ext.dataset.eq(dataset)&ext.Corerectal.notna());tag=f'external_mean_{kind}_{policy}_{dataset}'
                path=prediction(ext,ix,mean[ix],out/'predictions'/f'{tag}.csv.gz',panel='Regional5',method=f'{kind}_{policy}',role='external')
                complete(out,tag,[path],fits_averaged=10,selection='source only',**metrics(pd.read_csv(path)))
                log('EXTERNAL',tag,metrics(pd.read_csv(path)))

def report(out):
    frames=[]
    for p in sorted((out/'predictions').glob('*_test.csv.gz')):
        if p.name.startswith('external_'):continue
        frames.append(pd.read_csv(p))
    result=[]
    if frames:
        allq=pd.concat(frames,ignore_index=True)
        for (panel,method),q in allq.groupby(['panel','method']):
            complete_keys=len(q)==17153 and q.source_excel_row.is_unique and q.Participant.nunique()==40
            channels=read(HERE/'input_manifest.json')['panels'][panel]
            if method in ['EKF_Reconstructed','HybridCNNSimple']:channels=['head','HR']
            elif method=='Buller_HR':channels=['HR']
            result.append(dict(panel=panel,method=method,input_channels=';'.join(channels),complete=complete_keys,**metrics(q)))
        pd.DataFrame(result).to_csv(out/'source_summary.csv',index=False)
    external=[]
    for p in sorted((out/'predictions').glob('external_mean_*.csv.gz')):
        q=pd.read_csv(p);external.append(dict(method=q.method.iloc[0],dataset=q.dataset.iloc[0],**metrics(q)))
    if external:pd.DataFrame(external).to_csv(out/'external_summary.csv',index=False)

def smoke(args,d,splits,manifest,out):
    # Tiny numerical check only; separate file and no experiment predictions.
    s=splits[0];raw,filled,med,full,simple=physical_inputs(d,s);ii=rows(d,s);tr=ii['train'][:64]
    tests={}
    for method in eswa.NEURAL:
        x,base=eswa.inputs_for(method,filled,full,simple);wi=eswa.window_indices(d.trial.to_numpy(),5);xx=torch.tensor(x[wi[tr[:16]]],dtype=torch.float32,device=args.device)
        m=eswa.WindowNetwork(eswa.architecture(method),x.shape[1]).to(args.device);opt=torch.optim.Adam(m.parameters(),lr=.001);loss=m(xx).square().mean();loss.backward();opt.step();assert torch.isfinite(loss);tests[method]=float(loss.detach().cpu())
    x=pd.concat([h.features(g,['head','HR','Temp','Humidity']) for _,g in d.groupby('trial',sort=False)]).sort_index()
    for name in CLASSIC+['ExtraTrees']:
        model=classic_create(name,grids(name)[0],args.threads)
        if name in ('CatBoost','LightGBM','XGBoost'):model.set_params(**({'iterations':3} if name=='CatBoost' else {'n_estimators':3}))
        elif name=='ExtraTrees':model.set_params(model__n_estimators=3)
        model=classic_fit(name,model,x,d,tr,ii['validation'][:32]);assert np.isfinite(model.predict(x.iloc[ii['validation'][:32]])).all();tests[name]='small fit passed'
    for kind in ['CNN','GRU']:
        model=full_model(kind).to(args.device);loss=model(torch.randn(2,20,20,device=args.device)).square().mean();loss.backward();assert torch.isfinite(loss);tests[kind+'_full']='gradient passed'
    if args.pfn_smoke:
        from tabpfn.model_loading import save_fitted_tabpfn_model,load_fitted_tabpfn_model
        weight=pfn_weight(args);model=make_pfn(args,weight)
        take=ii['train'][np.linspace(0,len(ii['train'])-1,64,dtype=int)];query=ii['validation'][:8]
        model.fit(x.iloc[take],d.Corerectal.iloc[take].to_numpy())
        outputs={k:np.asarray(model.predict(x.iloc[query],output_type=k)) for k in ['mean','median']}
        assert all(np.isfinite(v).all() for v in outputs.values())
        path=out/'smoke_tabpfn.tabpfn_fit';save_fitted_tabpfn_model(model,path);del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        clone=load_fitted_tabpfn_model(path,device=args.device)
        errors={k:float(abs(v-clone.predict(x.iloc[query],output_type=k)).max()) for k,v in outputs.items()}
        assert max(errors.values())<5e-5,errors
        tests['TabPFN_3_5']=dict(context_rows=len(take),query_rows=len(query),replay_errors=errors,weight_sha256=sha(weight))
        del clone;gc.collect()
    save(out/'smoke.json',dict(status='passed',purpose='implementation check only, not scientific results',device=args.device,tests=tests))
    log('SMOKE PASSED')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','smoke','run','report'])
    parser.add_argument('--bundle-root',type=Path,default=HERE.parents[1]);parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--pfn-weight',type=Path);parser.add_argument('--panels',default='all');parser.add_argument('--stages',default='tables,physiology,windows,external')
    parser.add_argument('--pfn-smoke',action='store_true',help='Include actual tiny TabPFN fit/save/reload/predict in smoke')
    args=parser.parse_args()
    with threadpool_limits(args.threads):
        d,splits,manifest,out=setup(args)
        if args.command=='report':report(out);return
        if args.command in ('preflight','smoke'):preflight(args,d,splits,manifest,out)
        if args.command=='smoke':smoke(args,d,splits,manifest,out)
        if args.command=='run':
            if not (out/'preflight.json').exists():preflight(args,d,splits,manifest,out)
            stages=args.stages.split(',')
            for stage in stages:
                if stage=='tables':tables(args,d,splits,manifest,out)
                elif stage=='physiology':physiology(args,d,splits,out)
                elif stage=='windows':neural_windows(args,d,splits,out)
                elif stage=='external':external_train(args,d,splits,out);external_evaluate(args,d,splits,out)
                else:raise ValueError(stage)
                report(out)
            save(out/'requested_stages_complete.json',dict(status='complete',stages=stages,panels=args.panels,source_labels=17153,protocol_sha=sha(out/'protocol.json')))
            log('REQUESTED STAGES COMPLETE',stages)
if __name__=='__main__':main()
