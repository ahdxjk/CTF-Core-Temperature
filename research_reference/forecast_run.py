"""New direct CTF forecasts; original estimation and forecast artifacts remain read-only."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="4"
os.environ["TABPFN_DISABLE_TELEMETRY"]="1"
from pathlib import Path
import sys,json,time,gc,hashlib,argparse,contextlib
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch,joblib,psutil
from threadpoolctl import threadpool_limits
from strict_confirmation_data_v1 import load,read,save,sha
from prepare_multikernel_backbone_v1 import SPLITS,SEEDS,CHANNELS as ALL22
from run_raw_tcn_all_v1 import encoding
from channel_bank400_v1 import trial_features
import forecast_extension_v2 as f
from models import build as build_ours
HORIZONS=[5,15,30]
KINDS=["MSCA","HE","GRU","TCN"]
ROLES=["train","validation","test"]
META=["source_excel_row","Participant","Condition","trial","minute","phase_5"]
R3=ROOT/"outputs/revision_round3_20261108/forecast"
WEIGHT=ROOT/"external/TabPFN35/tabpfn-v3.5-20260909.safetensors"
for folder in ("records","predictions","models","checkpoints","history"):
    (OUT/folder).mkdir(parents=True,exist_ok=True)
def log(*a):print(time.strftime("%Y-%m-%d %H:%M:%S"),*a,flush=True)
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def runtime(mode):
    global LIMITER
    p=psutil.Process();cpus=sorted(p.cpu_affinity())
    p.cpu_affinity(cpus[4:8] if mode=="et" and len(cpus)>=8 else cpus[:4])
    LIMITER=threadpool_limits(4)
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="4"
    torch.set_num_threads(4)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if mode not in ("et","report"):
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.65)
        from tabpfn_memory_runtime_v1 import configure_pfn_chunks
        configure_pfn_chunks(65536)
def dataset():
    d=load()
    assert len(d)==17264 and d.Corerectal.notna().sum()==17153
    assert d.index.equals(pd.RangeIndex(len(d)))
    table=pd.concat([f.table_features(g) for _,g in d.groupby("trial",sort=False)]).sort_index()
    table=table.replace([np.inf,-np.inf],np.nan)
    assert table.shape==(17264,324)
    yy,targets,_,vv=f.labels(d)
    ys={h:yy[:,f.HORIZONS.index(h)] for h in HORIZONS}
    dst={h:targets[:,f.HORIZONS.index(h)] for h in HORIZONS}
    valid={h:vv[:,f.HORIZONS.index(h)] for h in HORIZONS}
    assert {h:int(valid[h].sum()) for h in HORIZONS}=={5:16389,15:14845,30:12534}
    return d,table,ys,dst,valid
def build(kind):
    if kind in ("MSCA","HE"):return build_ours(kind)
    from baseline_models import build as make
    return make(kind)
def pack(d,s):
    x,_,_,rows,ids,groups,_,sc,_=encoding(d,s,ALL22)
    slots=x.reshape(*x.shape[:2],22,4)
    msca=np.zeros((*slots.shape[:3],3),np.float32)
    enabled=np.array([c in f.CHANNELS for c in ALL22],bool)
    observed=slots[...,1].astype(bool)&enabled[None,None,:]
    msca[...,0]=np.where(observed,slots[...,0],0)
    msca[...,1]=observed
    msca[...,2]=enabled
    nchan=[ALL22.index(c) for c in f.CHANNELS]
    raw=slots[:,:,nchan,:].copy()
    he=np.zeros((*rows.shape,14,400),np.float32)
    for i in range(len(rows)):
        n=int((rows[i]>=0).sum())
        he[i,:n]=trial_features(raw[i,:n])
    safe=rows.clip(min=0)
    act=np.stack((d.workrest.eq(1).to_numpy()[safe],d.workrest.isin([1,2]).to_numpy()[safe]),-1).astype(np.float32)
    act[rows<0]=0
    return dict(MSCA=msca,HE=he,GRU=raw,TCN=raw,rows=rows,ids=ids,groups=groups,scale=sc,activity=act)
def frame(d,s,h,role,pred,ys,dst,valid):
    idx=np.flatnonzero(d.Participant.isin(s[role])&valid[h])
    q=d.iloc[idx][META].copy()
    q["target_excel_row"]=d.source_excel_row.iloc[dst[h][idx]].to_numpy()
    q["truth"]=ys[h][idx]
    q["prediction"]=np.asarray(pred)[idx]
    q["fold"]=s["fold"];q["horizon"]=h;q["panel"]="Neck14"
    assert np.isfinite(q.prediction).all()
    return q
def pmae(q):
    return float((q.prediction-q.truth).abs().groupby(q.Participant).mean().mean())
def atomic_torch(obj,path):
    tmp=path.with_suffix(path.suffix+".tmp");torch.save(obj,tmp);tmp.replace(path)
def record(tag):return OUT/"records"/(tag+".json")
def done(tag):
    p=record(tag)
    if not p.exists():return False
    z=read(p)
    assert z["protocol_sha"]==sha(OUT/"protocol.json"),tag
    for path,h in z["hashes"].items():assert sha(path)==h,path
    return True
def commit(tag,model,frames,extra):
    paths=[model]
    for role,q in frames.items():
        path=OUT/"predictions"/f"{tag}_{role}.csv.gz"
        q.to_csv(path,index=False);paths.append(path)
    save(record(tag),dict(status="passed",protocol_sha=sha(OUT/"protocol.json"),hashes={str(p):sha(p) for p in paths},**extra))
    log("DONE",tag,"seconds",round(extra.get("seconds",0),1))
def protocol():
    inputs=[ROOT/"outputs/strict_confirmation_v1/clean_raw.pkl",ROOT/"outputs/strict_confirmation_v1/audit.json",ROOT/"outputs/cpu_baselines_v1/splits.json",WEIGHT]
    code=[Path(__file__),OUT/"models.py",OUT/"baseline_models.py",ROOT/"forecast_extension_v2.py",ROOT/"run_raw_tcn_all_v1.py",ROOT/"multikernel_backbone_model_v1.py",ROOT/"channel_bank400_v1.py"]
    return dict(version="CTF-direct-future-20261113-v1",created="2026-11-13",splits=SPLITS,seeds=SEEDS,horizons=HORIZONS,panel="Neck14",channels=f.CHANNELS,
        data_hashes={str(p):sha(p) for p in inputs},code_hashes={str(p):sha(p) for p in code},
        targets="Separate direct model per horizon; future measured rectal target at t+h. Same valid origins as R3 NoPlan; eligibility uses target availability and in-trial activity annotations, not input features.",
        inputs="14 physiological sensor histories <=t plus current_work/current_activity_known only. No current/previous true core temperature, phase, trial identifier, future sensor values or future activity/plan values as predictors.",
        origins={"5":16389,"15":14845,"30":12534},
        neural="Existing MSCA MK357 and HE Direct400 encoders. MSCA 22 registry slots with only Neck14 enabled; HE14 channels*400 causal descriptors. Last linear readouts concat two current activity flags:90->1 /66->1. Independent horizons; fixed two-seed average within each family.",
        neural_training=dict(epochs=150,patience=20,batch_trials=8,optimizer="AdamW",lr=.001,weight_decay=.001,clip=1,loss="participant-macro MAE",selection="validation participant-MAE",target_scaling="training-role eligible future targets only",dtype="FP32"),
        controls="GRU64/2 layers and plain TCN64/8 blocks use the same 14-channel full causal histories encoded as last/fresh/age/ever plus current activity flags; same two seeds, training objective, early stopping and origins.",
        tabular="324 identical causal features as archived R3 CatBoost NoPlan. ET400 inherits frozen source-validation-selected Neck14 fold settings; refitted on eligible future targets with participant weights and train-only imputation. TabPFN local3.5 frozen weights,2 estimators,full training context,FP32CUDA,hostcache/query16; validation future-MAE selects mean/median; no sample weights in context.",
        fusion="Arithmetic mean of MSCA seed-mean, HE seed-mean, TabPFN, ExtraTrees; each family exactly .25. No test-selected weights, members, seeds or horizons.",
        fair_baseline="R3 CatBoost NoPlan reused only after exact participant/row/target/label identity checks. Planned models are not same-information comparators.",
        statistics="Participant-macro MAE/RMSE/Bias; paired participant bootstrap10000,seed20261113. All participant test folds pooled once. Reused explored partitions, not untouched external confirmation.",
        fits=dict(neural=120,CTF_neural=60,architecture_controls=60,PFN=15,ET=15),
        hardware=dict(python=sys.executable,torch=torch.__version__,GPU=torch.cuda.get_device_name(),GPU_memory_GB=torch.cuda.get_device_properties(0).total_memory/2**30,threads=4))
def check():
    p=read(OUT/"protocol.json")
    for group in ("data_hashes","code_hashes"):
        for name,h in p[group].items():assert sha(name)==h,("Frozen file changed",name)
    assert read(OUT/"preflight.json")["status"]=="passed"
    return p
def prepare(stuff):
    d,table,ys,dst,valid=stuff
    for s in SPLITS:
        assert [len(s[r]) for r in ROLES]==[24,8,8]
        assert len(set(sum([s[r] for r in ROLES],[])))==40
        for h in HORIZONS:
            for role in ("validation","test"):
                q=pd.read_csv(R3/"predictions"/f'Neck14_all_NoPlan_h{h}_f{s["fold"]}_{role}.csv.gz')
                idx=np.flatnonzero(d.Participant.isin(s[role])&valid[h])
                np.testing.assert_array_equal(q.source_excel_row,d.source_excel_row.iloc[idx])
                np.testing.assert_array_equal(q.target_excel_row,d.source_excel_row.iloc[dst[h][idx]])
                np.testing.assert_allclose(q.truth,ys[h][idx],atol=1e-10,rtol=0)
    # Features depend on the prefix only; target columns never enter table inputs.
    for _,g in list(d.groupby("trial",sort=False))[:3]:
        cut=min(25,len(g))
        altered=g.copy();altered.loc[g.index[cut:],f.CHANNELS]=999
        np.testing.assert_allclose(f.table_features(g).iloc[:cut],f.table_features(altered).iloc[:cut],equal_nan=True)
        altered=g.copy();altered["Corerectal"]=1234
        np.testing.assert_allclose(f.table_features(g),f.table_features(altered),equal_nan=True)
    p=pack(d,SPLITS[0]);checks={}
    train=d.Participant.isin(SPLITS[0]["train"])
    np.testing.assert_allclose(p["scale"]["mean"],d.loc[train,ALL22].mean().fillna(0))
    for kind in KINDS:
        torch.manual_seed(SEEDS[0])
        net=build(kind).cuda().eval()
        x=torch.tensor(p[kind][:2],device="cuda")
        act=torch.tensor(p["activity"][:2],device="cuda")
        cut=25
        with torch.no_grad():
            original=net(x,act)
            altered=x.clone();altered[:,cut:]=999
            changed=act.clone();changed[:,cut:]=1-changed[:,cut:]
            err=float((net(altered,changed)[:,:cut]-original[:,:cut]).abs().max())
            err2=float((net(x[:,:cut],act[:,:cut])-original[:,:cut]).abs().max())
        assert max(err,err2)<5e-5,(kind,err,err2)
        net.train();net(x,act).mean().backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in net.parameters())
        checks[kind]=dict(parameters=sum(v.numel() for v in net.parameters()),future_error=err,prefix_error=err2)
        del net,x,act;release()
    proto=protocol()
    if (OUT/"protocol.json").exists():assert read(OUT/"protocol.json")==proto
    else:save(OUT/"protocol.json",proto)
    save(OUT/"preflight.json",dict(status="passed",checks=checks,labels=17153,origins=proto["origins"],split_and_R3_origin_identity=True,prefix_and_target_isolation=True,train_only_scaling=True))
    log("PREPARED",checks)
def neural(d,s,h,kind,seed,p,ys,dst,valid):
    tag=f"{kind}_h{h}_s{seed}_f{s['fold']}"
    if done(tag):return
    started=time.monotonic();log("START",tag)
    rows=p["rows"];safe=rows.clip(min=0);present=rows>=0
    tr=np.flatnonzero(d.Participant.isin(s["train"])&valid[h])
    ym=float(np.mean(ys[h][tr]));sd=float(np.std(ys[h][tr],ddof=1));assert sd>0
    y=np.nan_to_num((ys[h][safe]-ym)/sd).astype(np.float32)
    active=valid[h][safe]&present
    weights=np.zeros(rows.shape,np.float32)
    for pid in s["train"]:
        mask=(p["ids"][:,None]==pid)&active
        assert mask.any();weights[mask]=1/(24*mask.sum())
    assert np.isclose(weights.sum(),1) and not weights[~active].any()
    torch.manual_seed(seed+s["fold"]);rng=np.random.RandomState(seed+s["fold"])
    net=build(kind).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    xx,aa,yy,ww=[torch.tensor(z,device="cuda") for z in (p[kind],p["activity"],y,weights)]
    def predict(role):
        net.eval();indices=p["groups"][role];result=np.full(len(d),np.nan)
        with torch.no_grad():
            for first in range(0,len(indices),8):
                ix=indices[first:first+8]
                pred=net(xx[ix],aa[ix]).cpu().numpy()*sd+ym
                rr=rows[ix];ok=rr>=0;result[rr[ok]]=pred[ok]
        return frame(d,s,h,role,result,ys,dst,valid)
    cp=OUT/"checkpoints"/(tag+".pt")
    epoch0=0;best=float("inf");bestepoch=-1;history=[];state=None
    if cp.exists():
        ck=torch.load(cp,map_location="cpu",weights_only=False)
        assert ck["protocol_sha"]==sha(OUT/"protocol.json")
        net.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        epoch0=ck["epoch"]+1;best=ck["best"];bestepoch=ck["bestepoch"];history=ck["history"];state=ck["best_state"]
        rng.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
        log("RESUME",tag,"epoch",epoch0)
    for epoch in range(epoch0,150):
        net.train();order=rng.permutation(p["groups"]["train"])
        for first in range(0,len(order),8):
            ix=order[first:first+8];opt.zero_grad(set_to_none=True)
            loss=((net(xx[ix],aa[ix])-yy[ix]).abs()*ww[ix]).sum()*len(order)/len(ix)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step()
        val=predict("validation");score=pmae(val);history.append(score)
        if score<best:
            best=score;bestepoch=epoch;state={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
        atomic_torch(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,bestepoch=bestepoch,history=history,best_state=state,numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state().cpu(),protocol_sha=sha(OUT/"protocol.json")),cp)
        if epoch==0 or (epoch+1)%10==0:log(tag,"epoch",epoch+1,"validation_MAE",round(score,6))
        if epoch-bestepoch>=20:break
    net.load_state_dict(state)
    frames={r:predict(r) for r in ("validation","test")}
    model=OUT/"models"/(tag+".pt")
    atomic_torch(dict(state=state,sensor_scale=p["scale"],target_scale=dict(mean=ym,std=sd),kind=kind,horizon=h,seed=seed,split=s),model)
    net.load_state_dict(torch.load(model,map_location="cpu",weights_only=False)["state"])
    replay=predict("test")
    err=float(np.max(np.abs(replay.prediction-frames["test"].prediction)));assert err<5e-5
    save(OUT/"history"/(tag+".json"),dict(validation_MAE=history,best_epoch=bestepoch+1))
    commit(tag,model,frames,dict(kind=kind,horizon=h,seed=seed,fold=s["fold"],best_epoch=bestepoch+1,best_validation_MAE=best,epochs=len(history),seconds=time.monotonic()-started,reload_error=err))
    del net,opt,xx,aa,yy,ww,state;release()
def table_fit(d,table,s,h,kind,ys,dst,valid):
    tag=f"{kind}_h{h}_f{s['fold']}"
    if done(tag):return
    started=time.monotonic();log("START",tag)
    ix={r:np.flatnonzero(d.Participant.isin(s[r])&valid[h]) for r in ROLES};tr=ix["train"]
    if kind=="ET":
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import ExtraTreesRegressor
        params=read(ROOT/"outputs/strict_confirmation_v1/protocol.json")["table_settings"][f"Neck14_f{s['fold']}"]["ET"]
        params["n_jobs"]=4
        net=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("model",ExtraTreesRegressor(**params))])
        pid=d.Participant.iloc[tr];w=1/pid.map(pid.value_counts()).to_numpy()
        net.fit(table.iloc[tr],ys[h][tr],model__sample_weight=w/w.mean())
        def predict(idx):return net.predict(table.iloc[idx])
        settings=dict(params=params,inherited_frozen_source_validation_settings=True)
    else:
        from tabpfn import TabPFNRegressor
        net=TabPFNRegressor(model_path=str(WEIGHT),device="cuda",n_estimators=2,n_preprocessing_jobs=4,random_state=20261030,fit_mode="fit_with_cache",inference_precision=torch.float32,memory_saving_mode=True,keep_cache_on_device=False,show_progress_bar=False)
        net.fit(table.iloc[tr],ys[h][tr])
        outputs={"mean":[],"median":[]}
        for first in range(0,len(ix["validation"]),16):
            z=net.predict(table.iloc[ix["validation"][first:first+16]],output_type="main")
            for key in outputs:outputs[key].append(z[key])
        errors={}
        for key,parts in outputs.items():
            pp=np.full(len(d),np.nan);pp[ix["validation"]]=np.concatenate(parts)
            errors[key]=pmae(frame(d,s,h,"validation",pp,ys,dst,valid))
        selection=min(("mean","median"),key=lambda k:errors[k])
        log(tag,"validation_output",selection,errors)
        def predict(idx):
            return np.concatenate([net.predict(table.iloc[idx[first:first+16]],output_type=selection) for first in range(0,len(idx),16)])
        settings=dict(output=selection,validation_MAE=errors,context_rows=len(tr),feature_count=table.shape[1],local_weight_sha=sha(WEIGHT))
    frames={}
    for role in ("validation","test"):
        pp=np.full(len(d),np.nan)
        pp[ix[role]]=np.concatenate(outputs[selection]) if kind=="PFN" and role=="validation" else predict(ix[role])
        frames[role]=frame(d,s,h,role,pp,ys,dst,valid)
    model=OUT/"models"/(tag+(".joblib" if kind=="ET" else ".tabpfn_fit"))
    if kind=="ET":joblib.dump(net,model,compress=3);net=joblib.load(model)
    else:
        from tabpfn.model_loading import save_fitted_tabpfn_model,load_fitted_tabpfn_model
        save_fitted_tabpfn_model(net,model);del net;release()
        net=load_fitted_tabpfn_model(model,device="cpu")
        net.keep_cache_on_device=False;net.executor_.keep_cache_on_device=False;net.memory_saving_mode=True;net.to("cuda")
    probe=ix["test"][:16];err=float(np.max(np.abs(predict(probe)-frames["test"].prediction.to_numpy()[:16])));assert err<1e-4
    commit(tag,model,frames,dict(kind=kind,horizon=h,fold=s["fold"],seconds=time.monotonic()-started,reload_probe_error=err,settings=settings))
    del net;release()
def report(partial=False):
    parts=[];coverage=[]
    for h in HORIZONS:
        for s in SPLITS:
            roles={}
            for kind in ("MSCA","HE","PFN","ET","GRU","TCN"):
                tags=[f"{kind}_h{h}_s{seed}_f{s['fold']}" for seed in SEEDS] if kind in KINDS else [f"{kind}_h{h}_f{s['fold']}"]
                if not all(done(tag) for tag in tags):continue
                qs=[pd.read_csv(OUT/"predictions"/f"{tag}_test.csv.gz").sort_values("source_excel_row").reset_index(drop=True) for tag in tags]
                q=qs[0].copy()
                for other in qs[1:]:
                    np.testing.assert_array_equal(q.source_excel_row,other.source_excel_row)
                    np.testing.assert_allclose(q.truth,other.truth,atol=1e-10,rtol=0)
                q["prediction"]=np.mean([z.prediction.to_numpy() for z in qs],axis=0)
                q["method"]=kind;roles[kind]=q;parts.append(q)
            if all(k in roles for k in ("MSCA","HE","PFN","ET")):
                q=roles["MSCA"].copy()
                for k in ("HE","PFN","ET"):
                    np.testing.assert_array_equal(q.source_excel_row,roles[k].source_excel_row)
                    np.testing.assert_allclose(q.truth,roles[k].truth,atol=1e-10,rtol=0)
                q["prediction"]=np.mean([roles[k].prediction.to_numpy() for k in ("MSCA","HE","PFN","ET")],axis=0)
                q["method"]="CTF_Direct";parts.append(q)
                cat=pd.read_csv(R3/"predictions"/f"Neck14_all_NoPlan_h{h}_f{s['fold']}_test.csv.gz").sort_values("source_excel_row").reset_index(drop=True)
                np.testing.assert_array_equal(q.source_excel_row,cat.source_excel_row)
                np.testing.assert_allclose(q.truth,cat.truth,atol=1e-10,rtol=0)
                cat["method"]="CatBoost_NoPlan";cat["horizon"]=h;cat["fold"]=s["fold"];parts.append(cat)
                coverage.append(dict(horizon=h,fold=s["fold"],people=q.Participant.nunique(),origins=len(q)))
    save(OUT/"progress.json",dict(records=len(list((OUT/"records").glob("*.json"))),expected_records=150,completed_CTF_folds=coverage))
    if not parts:return
    q=pd.concat(parts,ignore_index=True);q["error"]=q.prediction-q.truth;q["ae"]=q.error.abs();q["se"]=q.error**2
    q.to_csv(OUT/"predictions_all_partial.csv.gz",index=False)
    p=q.groupby(["horizon","method","Participant"],as_index=False).agg(MAE=("ae","mean"),MSE=("se","mean"),Bias=("error","mean"),origins=("ae","size"))
    p["RMSE"]=np.sqrt(p.MSE)
    summary=p.groupby(["horizon","method"],as_index=False).agg(MAE=("MAE","mean"),RMSE=("RMSE","mean"),Bias=("Bias","mean"),people=("Participant","size"),origins=("origins","sum"))
    summary.to_csv(OUT/"summary_partial.csv",index=False);p.to_csv(OUT/"participant_partial.csv",index=False)
    if len(list((OUT/"records").glob("*.json")))!=150:
        if not partial:raise RuntimeError("Incomplete experiment; partial summaries are not final.")
        return
    assert len(coverage)==15 and summary.people.eq(40).all()
    assert not q.duplicated(["horizon","method","source_excel_row"]).any()
    for h,g in q.groupby("horizon"):
        keys=[set(x.source_excel_row) for _,x in g.groupby("method")]
        assert all(x==keys[0] for x in keys)
        assert len(keys[0])=={5:16389,15:14845,30:12534}[h]
    summary.to_csv(OUT/"summary.csv",index=False);p.to_csv(OUT/"participant.csv",index=False);q.to_csv(OUT/"predictions_all.csv.gz",index=False)
    comparisons=[]
    for h,g in p.groupby("horizon"):
        wide=g.pivot(index="Participant",columns="method",values="MAE")
        for ref in wide.columns:
            if ref=="CTF_Direct":continue
            delta=(wide.CTF_Direct-wide[ref]).to_numpy();rng=np.random.default_rng(20261113)
            lo,hi=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975])
            comparisons.append(dict(horizon=h,reference=ref,delta_MAE=float(delta.mean()),low=float(lo),high=float(hi),people=len(delta)))
    pd.DataFrame(comparisons).to_csv(OUT/"paired_ci.csv",index=False)
    save(OUT/"verification.json",dict(status="passed",records=150,matched_origins=True,participant_split_isolation=True,CTF_family_weights=[.25]*4,old_results_overwritten=False,protocol_sha=sha(OUT/"protocol.json")))
    (OUT/"report.html").write_text('<meta charset="utf-8"><h1>New direct four-family forecasts</h1><p>Neck14, current activity, no future plan; participant-held-out five folds.</p>'+summary.to_html(index=False)+pd.DataFrame(comparisons).to_html(index=False),encoding="utf-8")
    log("COMPLETE",summary.to_string(index=False))
def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["prepare","gpu","et","report"])
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args();runtime(args.mode)
    if args.mode=="report":check();report(args.partial);return
    stuff=dataset()
    if args.mode=="prepare":prepare(stuff);return
    check();d,table,ys,dst,valid=stuff
    import msvcrt
    with (OUT/(args.mode+".lock")).open("a+b") as lock:
        lock.write(b"0");lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        for s in SPLITS:
            p=pack(d,s) if args.mode=="gpu" else None
            for h in HORIZONS:
                if args.mode=="gpu":
                    for kind in KINDS:
                        for seed in SEEDS:neural(d,s,h,kind,seed,p,ys,dst,valid)
                    table_fit(d,table,s,h,"PFN",ys,dst,valid)
                else:table_fit(d,table,s,h,"ET",ys,dst,valid)
                report(partial=True)
            del p;release()
        report(partial=True)
if __name__=="__main__":main()

