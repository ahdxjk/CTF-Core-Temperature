"""Clean-source Regional5 neural retraining; external data evaluated only after training.

Root reviews the CPU preflight and draft before explicitly freezing the protocol.
No legacy files, clean-source audits, source tables or fusion weights are changed.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse
import contextlib
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from threadpoolctl import threadpool_limits
from prepare_multikernel_backbone_v1 import CHANNELS,SPLITS,SEEDS
from run_raw_tcn_all_v1 import encoding
from channel_bank400_v1 import trial_features
from regional_neural_transfer_v2_model import build_setting,input_kind,structure

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/regional_neural_transfer_v2'
MOD=Path('REPLACE_WITH_MODEL_DIR/regional_neural_transfer_v2')
COLS=['chest','upper arm','thigh','calf','HR']
SETTINGS=['MSCA_FreshNative','HE_Native','TCN_Native','GRU_Native',
          'MSCA_FreshRateMix','MSCA_HeldAgeRateMix','HE_Sparse5','HE_RateMix']
POLICIES=['Native','Robust']
SCENARIOS=['Native','SourceSparse5','SourceBlock26']
ROLES=['source_test','Hotwater','Brighton']
VIEWS=(0,2,3,5)
KEYS=['dataset','Participant','trial','minute','source_excel_row','Corerectal','phase_5','Condition']
SOURCE_ROWS,SOURCE_LABELS=17264,17153
DEVICE='cpu'
_limit=None


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def runtime(device='cpu'):
    global DEVICE,_limit
    DEVICE=device;psutil.Process().cpu_affinity([8,9]);_limit=threadpool_limits(2)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:assert torch.get_num_interop_threads()==1
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':
        assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.65)
    for p in (OUT,OUT/'records',OUT/'predictions',OUT/'evaluations',OUT/'cache_records',MOD,MOD/'cache'):
        p.mkdir(parents=True,exist_ok=True)
    return dict(device=device,cpu_threads=2,cpu_affinity=psutil.Process().cpu_affinity(),pid=os.getpid())
def log(*args):
    message=time.strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args));print(message,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.log').open('a',encoding='utf8') as f:f.write(message+'\n')
def torch_save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)


def load_source(clean_loader=None):
    """No external dataset/labels are loaded by draft, preflight, prepare or train."""
    if clean_loader is None:
        from regional_causal_inputs_v2 import load_source_clean
        clean_loader=load_source_clean
    d=clean_loader().copy()
    assert len(d)==SOURCE_ROWS and d.Corerectal.notna().sum()==SOURCE_LABELS
    assert d.index.equals(pd.RangeIndex(SOURCE_ROWS)) and d.source_excel_row.is_unique
    assert d.Participant.nunique()==40 and d.trial.nunique()==155
    assert d.groupby('trial').minute.diff().dropna().eq(1).all()
    d['dataset']='PROSPIE';d['trial']='prospie_'+d.trial.astype(str)
    return d


def clean_source_signature(d):
    # Hash data values/masks, including target availability; no inferred labels.
    cols=['source_excel_row','Participant','trial','minute','Corerectal']+COLS
    values=pd.util.hash_pandas_object(d[cols],index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


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


def msca_input(raw,kind):
    width=5 if kind=='HeldAge' else 3
    a=np.zeros((*raw.shape[:2],22,width),np.float32)
    for j,c in enumerate(COLS):
        k=CHANNELS.index(c);z,fresh,age,ever=[raw[:,:,j,t] for t in range(4)]
        if kind=='Fresh':
            a[:,:,k,0]=np.where(fresh>0,z,0);a[:,:,k,1]=fresh;a[:,:,k,2]=1
        else:
            a[:,:,k,0]=np.where(ever>0,z,0);a[:,:,k,1]=fresh
            a[:,:,k,2]=np.where(ever>0,np.log1p(age*60),0)
            a[:,:,k,3]=ever;a[:,:,k,4]=1
    return a


def input_from_raw(raw,kind):
    if kind in ('Fresh','HeldAge'):return msca_input(raw,kind)
    if kind=='Raw4':return raw.reshape(*raw.shape[:2],20)
    raise ValueError('HE requires recomputation of all400 features from each real trial')


def parameter_contract(d):
    from regional_causal_inputs_v2 import contract as clean_contract
    cleaning_contract=clean_contract()
    files=[Path(__file__),ROOT/'regional_neural_transfer_v2_model.py',ROOT/'regional_causal_inputs_v2.py',
        ROOT/'multikernel_backbone_model_v1.py',ROOT/'channel_bank400_v1.py',ROOT/'fair_peer_controls_v2.py',
        ROOT/'run_raw_tcn_all_v1.py',ROOT/'prepare_multikernel_backbone_v1.py',
        ROOT/'load_brighton_publication_v1.py',ROOT/'external_transfer_v1_data.py',
        ROOT/'outputs/raw_tcn_all_v1/raw.pkl',ROOT/'outputs/cpu_baselines_v1/splits.json',
        ROOT/'outputs/regional_input_causality_audit_v2/audit.json',
        ROOT/'outputs/regional_input_causality_audit_v2/mask_cells.csv']
    reporter=ROOT/'report_regional_neural_transfer_v2.py'
    if reporter.exists():files.append(reporter)
    return dict(version='regional_neural_transfer_v2',hashes={str(p):sha(p) for p in files},
        source_signature=clean_source_signature(d),source_rows=SOURCE_ROWS,source_actual_labels=SOURCE_LABELS,
        clean_input_contract=cleaning_contract,
        cleaning='Only unified audited clean loader. Mask formula pseudo-observations in source inputs and non-measured core labels; keep every17264 timestamp and all genuine values. This repairs data interfaces, not a target-value alteration.',
        panel='Regional5',channels=COLS,channel_slots=CHANNELS,settings=SETTINGS,seeds=SEEDS,splits=SPLITS,
        expected_fits=80,expected_checkpoints=160,policies=POLICIES,scenarios=SCENARIOS,
        new_architecture='HeldAge MSCA replaces only66->88 stem with110->88; same88 width, depthwise3/5/7,4head causal attention and per-time LayerNorm. Other settings use immutable MK357/Direct400/TCN64/GRU2x64.',
        held_slots=['held_z','fresh_actual_observation','log1p(age_in_minutes)','ever_observed','installed'],
        held_missing='Before first measurement: held_z=age=fresh=ever=0; installed fixed1. Last observation held causally; holding never changes fresh mask. OriginalHE/raw4 age remains hours, as in legacy encoding.',
        augmentation='Native settings alwaysNative. Other settings independently per trial and epoch choose .5Native/.5sparse. Sparse5 uses interval5; RateMix chooses one common interval2/3/5 uniformly per augmented trial. Each channel retains first real value then observations >=interval actual minutes apart. Native5min HR preserved. No interpolation/target synthesis.',
        features='HE all400 summaries rebuilt from each augmented observation view, not augmented old summary tensors. CacheNative/2/3/5 source views only. Whole original trial and all actual labels retained.',
        normalization='Fit only clean24 source training people before augmentation; source mean/std and target scale frozen forval/test/external. No external data loaded during training.',
        training={'epochs':150,'patience':20,'stop_on':'Robust=.5Native+.5Sparse5 validation personMAE',
            'checkpoints':'On same trajectory save lowestNative and lowestRobust; Native is not a separate early-stop trajectory.',
            'optimizer':'AdamW','lr':.001,'weight_decay':.001,'clip':1.,'batch_original_trials':8,
            'loss':'sum of absolute standardized errors with fixed personweights 1/(24*person_label_count), multiplied by n_train_trials/batch_trials',
            'rng':'seed+fold optimizer order; independent seed+fold+740000 augmentation generator; full RNG/optimizer state checkpointed every epoch'},
        evaluation='Only afterall80 fits. Outer source testNative/Sparse5/26min-block reset; Hotwater/Brighton onlyNative. Noexternalcheckpoint/weight/seedselection. Allpolicy outputs saved, fixed2seed means and5sourcefold external means handled byreport.',
        historical_references='Old dirty-trained models and fusion weights are historical only. New clean PFN/ET are coordinated separately; do not claim strict ranking with dirty references.',
        runtime={'device':'CUDA FP32, TF32off, allocator65% when authorized','cpu_threads':2,'cpu_affinity':[8,9]},
        model_storage=str(MOD),reporter_present=reporter.exists(),
        limitations='Repeated inspected domains and source splits; exploratory, uncorrected multiplicity; different settings need not run equal epochs/FLOPs. Regional sensor mapping, missingness and task/domain shift remain.')


def draft():
    d=load_source();pr=parameter_contract(d);save(OUT/'protocol_draft.json',pr);log('DRAFT READY');return d,pr


def assert_frozen(d=None):
    d=load_source() if d is None else d
    assert read(OUT/'protocol.json')==parameter_contract(d),'Protocol/code/clean-source drift'
    assert read(OUT/'preflight.json')['status']=='passed'
    return d


def preflight():
    assert DEVICE=='cpu','CPU preflight only; GPU check is a separate explicit command'
    d,pr=draft();s=SPLITS[0];arr,y,v,rows,ids,groups,w,scale,_=encoding(d,s,COLS)
    assert v.sum()==SOURCE_LABELS and (rows>=0).sum()==SOURCE_ROWS and np.isclose(w.sum(),1)
    assert set(d.loc[d.Participant.isin(s['train']),'Participant'])==set(s['train'])
    changed=d.copy();outside=~d.Participant.isin(s['train'])
    changed.loc[outside,COLS]=12345.;changed.loc[outside,'Corerectal']=-100
    assert encoding(changed,s,COLS)[7]==scale
    g=next(g for _,g in d.groupby('trial',sort=False))
    checks={};he_views={}
    for interval in VIEWS:
        sparse=sparsify(g,interval);raw=encode_trial(sparse,scale);he=trial_features(raw);he_views[interval]=he
        for cut in (1,7,17,min(31,len(g))):
            pd.testing.assert_frame_equal(sparsify(g,interval).iloc[:cut],sparsify(g.iloc[:cut],interval))
            rebuilt=encode_trial(sparsify(g.iloc[:cut],interval),scale)
            np.testing.assert_allclose(raw[:cut],rebuilt,atol=1e-6)
            np.testing.assert_allclose(he[:cut],trial_features(rebuilt),atol=1e-6)
        assert np.array_equal(sparse.Corerectal.notna(),g.Corerectal.notna())
        np.testing.assert_allclose(sparse.Corerectal,g.Corerectal,equal_nan=True)
        for c in COLS:
            kept=sparse.index[sparse[c].notna()];assert set(kept)<=set(g.index[g[c].notna()])
            if len(kept):assert kept[0]==g.index[g[c].notna()][0]
            if interval and len(kept)>1:assert (np.diff(sparse.loc[kept,'minute'])>=interval).all()
        checks[str(interval)]={'observations':int(sparse[COLS].notna().sum().sum()),'HE_prefix':True}
    toy=g.iloc[:21].copy();toy['HR']=np.nan;toy.loc[toy.index[::5],'HR']=np.arange(len(toy.iloc[::5]))+60
    for interval in (2,3,5):np.testing.assert_allclose(sparsify(toy,interval).HR,toy.HR,equal_nan=True)
    toy2=g.iloc[:15].copy();toy2.loc[:,COLS]=np.nan;toy2.loc[toy2.index[4],COLS]=g[COLS].median().fillna(30).to_numpy()
    raw=encode_trial(toy2,scale);assert (raw[:4,:,[0,1,2,3]]==0).all()
    held=msca_input(raw[None],'HeldAge');idx=CHANNELS.index(COLS[0])
    assert held[0,3,idx,0]==0 and held[0,3,idx,3]==0 and held[0,3,idx,4]==1
    assert held[0,7,idx,1]==0 and held[0,7,idx,3]==1
    np.testing.assert_allclose(held[0,7,idx,2],np.log1p(3),atol=1e-6)
    structures=[]
    # Same downstream layers/shapes: the held-age model changes the stem only.
    freshmodel=build_setting('MSCA_FreshNative');heldmodel=build_setting('MSCA_HeldAgeRateMix')
    freshshapes={k:list(v.shape) for k,v in freshmodel.state_dict().items() if not k.startswith('sensor_mix.')}
    heldshapes={k:list(v.shape) for k,v in heldmodel.state_dict().items() if not k.startswith('sensor_mix.')}
    assert freshshapes==heldshapes
    del freshmodel,heldmodel
    for setting in SETTINGS:
        torch.manual_seed(20261106);m=build_setting(setting).eval();kind=input_kind(setting)
        raw=encode_trial(g.iloc[:35],scale)
        xx=trial_features(raw)[None] if kind=='HE' else input_from_raw(raw[None],kind)
        x=torch.tensor(xx);future=x.clone();future[:,17:]+=10
        with torch.no_grad():
            a=m(x);b=m(future);c=m(x[:,:17])
            assert torch.allclose(a[:,:17],b[:,:17],atol=2e-5,rtol=1e-5),setting
            assert torch.allclose(a[:,:17],c,atol=2e-5,rtol=1e-5),setting
        m.train();m(x).sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        structures.append(structure(setting,m))
    # Re-encoded native values must reproduce immutable encoder semantics exactly.
    firstrows=rows[0][rows[0]>=0];native=encode_trial(d.iloc[firstrows],scale)
    np.testing.assert_allclose(native,arr[0,:len(firstrows)].reshape(len(firstrows),5,4),atol=1e-6)
    assert not np.array_equal(he_views[0],he_views[5]),'Sparse HE must actually recompute all views'
    # Observed names, targets and shape are hashed; no source labels are changed in augmentation.
    save(OUT/'preflight.json',dict(status='passed',device='cpu',structures=structures,
        source_rows=len(d),source_labels=int(d.Corerectal.notna().sum()),source_signature=clean_source_signature(d),
        draft_sha=sha(OUT/'protocol_draft.json'),future_prefix=True,gradients=True,sparse_prefix=True,
        sparse5_HR_preserved=True,held_age_minutes=True,no_future_fill=True,HE_recomputed_and_prefix=True,
        source_scaler_isolation=True,held_only_stem_changed=True,person_weight_sum=float(w.sum()),observation_checks=checks))
    log('CPU PREFLIGHT PASSED; ROOT REVIEW AND EXPLICIT FREEZE REQUIRED')


def freeze():
    d=load_source();pr=parameter_contract(d);pf=read(OUT/'preflight.json')
    assert pf['status']=='passed' and pf['draft_sha']==sha(OUT/'protocol_draft.json')
    assert read(OUT/'protocol_draft.json')==pr and pr['reporter_present'],'Reporter must exist before final CPU check/freeze'
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==pr
    else:save(OUT/'protocol.json',pr)
    log('PROTOCOL FROZEN')


def cache_fold(d,s):
    f=s['fold'];record=OUT/'cache_records'/f'fold{f}.json';dest=MOD/'cache'/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
    if record.exists():
        rec=read(record);assert rec['protocol_sha']==sha(OUT/'protocol.json')
        for p,h in rec['hashes'].items():assert sha(p)==h
        return rec
    x,y,v,rows,ids,groups,w,scale,_=encoding(d,s,COLS);n,t=rows.shape
    rawpath=dest/'raw.npy';hepath=dest/'he.npy';meta=dest/'metadata.npz'
    raw=np.lib.format.open_memmap(str(rawpath)+'.tmp',mode='w+',dtype='float32',shape=(4,n,t,5,4))
    he=np.lib.format.open_memmap(str(hepath)+'.tmp',mode='w+',dtype='float32',shape=(4,n,t,5,400))
    raw[:]=0;he[:]=0
    for vi,interval in enumerate(VIEWS):
        for i,rr in enumerate(rows):
            rr=rr[rr>=0];g=sparsify(d.iloc[rr],interval);a=encode_trial(g,scale)
            raw[vi,i,:len(rr)]=a;he[vi,i,:len(rr)]=trial_features(a)
        log('CACHE VIEW COMPLETE',f,interval)
    raw.flush();he.flush();del raw,he;os.replace(str(rawpath)+'.tmp',rawpath);os.replace(str(hepath)+'.tmp',hepath)
    np.savez_compressed(meta,y=y,valid=v,rows=rows,ids=ids,weights=w)
    rec=dict(status='passed',fold=f,split=s,scale=scale,groups={k:np.asarray(z).tolist() for k,z in groups.items()},
        raw_path=str(rawpath),he_path=str(hepath),metadata=str(meta),views=list(VIEWS),
        shape=[4,n,t],protocol_sha=sha(OUT/'protocol.json'),hashes={str(p):sha(p) for p in (rawpath,hepath,meta)})
    save(record,rec);return rec


class CachedInputs:
    def __init__(self,rec):
        self.rec=rec;self.raw=np.load(rec['raw_path'],mmap_mode='r');self.he=np.load(rec['he_path'],mmap_mode='r');q=np.load(rec['metadata'])
        self.y=q['y'];self.weights=q['weights'];self.rows=q['rows'];self.valid=q['valid'];self.groups={k:np.asarray(v) for k,v in rec['groups'].items()}
    def batch(self,indices,view_ids,kind):
        indices=np.asarray(indices);view_ids=np.asarray(view_ids)
        a=np.asarray(self.he[view_ids,indices]) if kind=='HE' else input_from_raw(np.asarray(self.raw[view_ids,indices]),kind)
        return torch.tensor(a,device=DEVICE)


def view_choices(setting,rng,n):
    # Native consumes the same independent augmentation RNG schedule for comparability.
    use=rng.random(n)<.5;rates=rng.integers(1,4,n)
    if setting.endswith('Native'):return np.zeros(n,dtype=int)
    if setting.endswith('Sparse5'):return np.where(use,3,0)
    assert setting.endswith('RateMix');return np.where(use,rates,0)


@torch.inference_mode()
def cached_predict(model,cache,indices,view,kind,scale):
    model.eval();pred=[];rr=[]
    for start in range(0,len(indices),8):
        ix=np.asarray(indices[start:start+8]);x=cache.batch(ix,np.full(len(ix),view),kind)
        pred.append(model(x).cpu().numpy()*scale['ystd']+scale['ymean']);rr.append(cache.rows[ix])
    rows=np.concatenate(rr).ravel();p=np.concatenate(pred).ravel();ok=rows>=0
    return pd.Series(p[ok],index=rows[ok]).sort_index()


def person_mae(d,p):
    ix=p.index;good=d.Corerectal.iloc[ix].notna().to_numpy();ii=ix[good];e=abs(p.to_numpy()[good]-d.Corerectal.iloc[ii].to_numpy())
    return float(pd.Series(e,index=d.Participant.iloc[ii]).groupby(level=0).mean().mean())


def fit_tag(setting,fold,seed):return f'{setting}_f{fold}_s{seed}'


def checked_record(path):
    rec=read(path);assert rec['status']=='passed'
    for p,h in rec['hashes'].items():assert sha(p)==h
    return rec


def train_fit(d,s,setting,seed,cache):
    f=s['fold'];tag=fit_tag(setting,f,seed);record=OUT/'records'/f'{tag}.json'
    if record.exists():checked_record(record);log('SKIP FIT',tag);return
    kind=input_kind(setting);scale=cache.rec['scale'];tr=cache.groups['train'];va=cache.groups['validation']
    torch.manual_seed(seed+f)
    if DEVICE=='cuda':torch.cuda.manual_seed_all(seed+f)
    model=build_setting(setting).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    order_rng=np.random.default_rng(seed+f);aug_rng=np.random.default_rng(seed+f+740000)
    resumed=MOD/(tag+'_resume.pt');history=[];best={p:float('inf') for p in POLICIES};infos={};bestepoch=0;start_epoch=1;updates=0
    if resumed.exists():
        q=torch.load(resumed,map_location='cpu',weights_only=False);assert q['protocol_sha']==sha(OUT/'protocol.json')
        model.load_state_dict(q['state']);opt.load_state_dict(q['optimizer'])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(DEVICE)
        history=q['history'];best=q['best'];infos=q['infos'];bestepoch=q['robust_bestepoch'];updates=q['updates'];start_epoch=q['epoch']+1
        order_rng.bit_generator.state=q['order_rng'];aug_rng.bit_generator.state=q['aug_rng'];torch.set_rng_state(q['torch_rng'])
        if DEVICE=='cuda':torch.cuda.set_rng_state_all(q['cuda_rng'])
        log('RESUME FIT',tag,'epoch',start_epoch)
    t0=time.perf_counter()
    # A stop boundary can be checkpointed just before an interruption. In that
    # case only replay/finalize; do not add another epoch to the frozen budget.
    stopped=bool(history and history[-1]['epoch']-bestepoch>=20)
    for epoch in ([] if stopped else range(start_epoch,151)):
        model.train();order=order_rng.permutation(tr)
        for st in range(0,len(order),8):
            batch=order[st:st+8];view=view_choices(setting,aug_rng,len(batch));xx=cache.batch(batch,view,kind)
            yy=torch.tensor(cache.y[batch],device=DEVICE);ww=torch.tensor(cache.weights[batch],device=DEVICE)
            opt.zero_grad(set_to_none=True);loss=((model(xx)-yy).abs()*ww).sum()*len(tr)/len(batch)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
        native=person_mae(d,cached_predict(model,cache,va,0,kind,scale));sparse=person_mae(d,cached_predict(model,cache,va,3,kind,scale));robust=.5*(native+sparse)
        history.append(dict(epoch=epoch,native_MAE=native,sparse5_MAE=sparse,robust_MAE=robust,updates=updates))
        for policy,value in [('Native',native),('Robust',robust)]:
            if value<best[policy]-1e-12:
                best[policy]=value
                if policy=='Robust':bestepoch=epoch
                path=MOD/(tag+'_'+policy+'.pt');state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch_save(path,dict(state=state,setting=setting,seed=seed,fold=f,split=s,scale=scale,channels=COLS,
                    policy=policy,bestepoch=epoch,protocol_sha=sha(OUT/'protocol.json')))
                infos[policy]=dict(path=str(path),sha256=sha(path),bestepoch=epoch,native_MAE=native,sparse5_MAE=sparse,robust_MAE=robust)
        torch_save(resumed,dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
            epoch=epoch,history=history,best=best,infos=infos,robust_bestepoch=bestepoch,updates=updates,
            order_rng=order_rng.bit_generator.state,aug_rng=aug_rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else [],protocol_sha=sha(OUT/'protocol.json')))
        if epoch==1 or epoch%10==0:log(tag,'epoch',epoch,'native',round(native,6),'sparse5',round(sparse,6),'robust',round(robust,6))
        if epoch-bestepoch>=20:break
    replay={}
    for policy in POLICIES:
        info=infos[policy];ck=torch.load(info['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
        # Full validation-source replay, both policies. No outer-test or external prediction here.
        p1=cached_predict(model,cache,va,0,kind,scale);p2=cached_predict(model,cache,va,3,kind,scale)
        clone=build_setting(setting).to(DEVICE);clone.load_state_dict(torch.load(info['path'],map_location='cpu',weights_only=False)['state'])
        e=max(float(abs(p1-cached_predict(clone,cache,va,0,kind,scale)).max()),float(abs(p2-cached_predict(clone,cache,va,3,kind,scale)).max()))
        assert e<5e-5 and abs(person_mae(d,p1)-info['native_MAE'])<5e-6 and abs(person_mae(d,p2)-info['sparse5_MAE'])<5e-6
        replay[policy]=e;del clone
    rec=dict(status='passed',setting=setting,fold=f,seed=seed,split=s,scale=scale,models=infos,
        hashes={x['path']:x['sha256'] for x in infos.values()},history=history,optimizer_steps=updates,epochs_run=len(history),
        replay_errors=replay,protocol_sha=sha(OUT/'protocol.json'),source_rows=SOURCE_ROWS,source_labels=SOURCE_LABELS,
        training_only_source=True,seconds=time.perf_counter()-t0,structure=structure(setting,model))
    save(record,rec);log('FIT COMPLETE',tag,'Native',infos['Native']['bestepoch'],'Robust',infos['Robust']['bestepoch'])
    del model,opt;gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()


def train(folds=None,settings=None,seeds=None):
    d=assert_frozen();folds=range(5) if folds is None else folds;settings=SETTINGS if settings is None else settings;seeds=SEEDS if seeds is None else seeds
    for f in folds:
        s=SPLITS[f];rec=cache_fold(d,s);cache=CachedInputs(rec)
        for setting in settings:
            for seed in seeds:train_fit(d,s,setting,seed,cache)
        del cache;gc.collect()
    expected=[OUT/'records'/f'{fit_tag(k,f,seed)}.json' for f in range(5) for k in SETTINGS for seed in SEEDS]
    if all(p.exists() for p in expected):
        for p in expected:checked_record(p)
        save(OUT/'training_complete.json',dict(status='complete',fits=80,checkpoints=160,source_labels=SOURCE_LABELS,
            protocol_sha=sha(OUT/'protocol.json'),external_datasets_loaded=False))
        log('ALL80 TRAINING TRAJECTORIES COMPLETE; EXTERNAL EVALUATION NOW ALLOWED')


def load_evaluation_data():
    # Deliberately late imports: training never opens external measurement/target data.
    from regional_causal_inputs_v2 import load_combined_clean
    d=load_combined_clean()
    assert d.source_excel_row.is_unique and d.groupby('trial').Participant.nunique().eq(1).all()
    assert [int(d[d.dataset.eq(name)].Corerectal.notna().sum()) for name in ['PROSPIE','Hotwater','Brighton']]==[SOURCE_LABELS,180,360]
    return d


def eval_frames(d,s,role,scenario):
    selected=d[d.Participant.isin(s['test']) & d.dataset.eq('PROSPIE')] if role=='source_test' else d[d.dataset.eq(role)]
    frames=[]
    for _,g in selected.groupby('trial',sort=False):
        if scenario=='SourceSparse5':g=sparsify(g,5)
        if scenario=='SourceBlock26':frames.extend(g.iloc[k:k+26] for k in range(0,len(g),26))
        else:frames.append(g)
    assert scenario=='Native' or role=='source_test'
    return frames


@torch.inference_mode()
def predict_frames(model,frames,scale,kind):
    model.eval();outputs=[]
    for start in range(0,len(frames),8):
        gs=frames[start:start+8];length=max(len(g) for g in gs);raw=np.zeros((len(gs),length,5,4),np.float32)
        for i,g in enumerate(gs):raw[i,:len(g)]=encode_trial(g,scale)
        if kind=='HE':
            a=np.zeros((len(gs),length,5,400),np.float32)
            for i,g in enumerate(gs):a[i,:len(g)]=trial_features(raw[i,:len(g)])
        else:a=input_from_raw(raw,kind)
        p=model(torch.tensor(a,device=DEVICE)).cpu().numpy()*scale['ystd']+scale['ymean']
        outputs.extend(pd.Series(pp[:len(g)],index=g.index) for g,pp in zip(gs,p))
    return pd.concat(outputs).sort_index()


def evaluate(folds=None,settings=None,seeds=None):
    assert_frozen();complete=read(OUT/'training_complete.json');assert complete['status']=='complete' and complete['fits']==80
    d=load_evaluation_data();folds=range(5) if folds is None else folds;settings=SETTINGS if settings is None else settings;seeds=SEEDS if seeds is None else seeds
    for f in folds:
        for setting in settings:
            kind=input_kind(setting)
            for seed in seeds:
                tag=fit_tag(setting,f,seed);rec=checked_record(OUT/'records'/f'{tag}.json');s=rec['split'];scale=rec['scale']
                for policy in POLICIES:
                    marker=OUT/'evaluations'/f'{tag}_{policy}.json'
                    if marker.exists():checked_record(marker);continue
                    path=rec['models'][policy]['path'];ck=torch.load(path,map_location='cpu',weights_only=False)
                    assert ck['split']==s and ck['scale']==scale and ck['setting']==setting and ck['policy']==policy
                    model=build_setting(setting).to(DEVICE);model.load_state_dict(ck['state']);clone=build_setting(setting).to(DEVICE)
                    clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);paths=[];errors={};counts={}
                    for role in ROLES:
                        for scenario in SCENARIOS if role=='source_test' else ['Native']:
                            gs=eval_frames(d,s,role,scenario);p=predict_frames(model,gs,scale,kind);again=predict_frames(clone,gs,scale,kind)
                            error=float(abs(p-again).max());assert error<5e-5
                            ix=p.index[d.Corerectal.loc[p.index].notna()];q=d.loc[ix,KEYS].copy();q['prediction']=p.loc[ix].to_numpy()
                            assert np.isfinite(q.prediction).all()
                            for name,value in dict(fold=f,setting=setting,policy=policy,seed=seed,role=role,scenario=scenario).items():q[name]=value
                            file=OUT/'predictions'/f'{tag}_{policy}_{role}_{scenario}.csv.gz';q.to_csv(file,index=False,compression='gzip')
                            paths.append(file);errors[role+'_'+scenario]=error;counts[role+'_'+scenario]=len(q)
                    save(marker,dict(status='passed',setting=setting,fold=f,seed=seed,policy=policy,model_sha=sha(path),
                        hashes={str(p):sha(p) for p in paths},replay_errors=errors,label_counts=counts,protocol_sha=sha(OUT/'protocol.json')))
                    log('EVALUATION COMPLETE',tag,policy);del model,clone;gc.collect()
                    if DEVICE=='cuda':torch.cuda.empty_cache()
    expected=[OUT/'evaluations'/f'{fit_tag(k,f,seed)}_{p}.json' for f in range(5) for k in SETTINGS for seed in SEEDS for p in POLICIES]
    if all(p.exists() for p in expected):
        for p in expected:checked_record(p)
        save(OUT/'evaluation_complete.json',dict(status='complete',checkpoints=160,records=160,source_labels=SOURCE_LABELS,Hotwater=180,Brighton=360))
        log('ALL160 CHECKPOINT EVALUATIONS COMPLETE')


def gpu_preflight():
    assert DEVICE=='cuda','Explicit --device cuda required'
    d=assert_frozen();s=SPLITS[0];scale=encoding(d,s,COLS)[7];g=next(g for _,g in d.groupby('trial',sort=False));results=[]
    for setting in SETTINGS:
        torch.manual_seed(20261106);m=build_setting(setting);kind=input_kind(setting);raw=encode_trial(g.iloc[:35],scale)
        x=trial_features(raw)[None] if kind=='HE' else input_from_raw(raw[None],kind)
        m.eval()
        with torch.no_grad():cpu=m(torch.tensor(x)).numpy();m=m.cuda();cuda=m(torch.tensor(x,device='cuda')).cpu().numpy()
        err=float(np.max(abs(cpu-cuda)));assert err<1e-4
        m.train();m(torch.tensor(x,device='cuda')).sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        results.append(dict(setting=setting,CPU_CUDA_error=err));del m;torch.cuda.empty_cache()
    save(OUT/'gpu_preflight.json',dict(status='passed',results=results,device=torch.cuda.get_device_name(0)))
    log('GPU PREFLIGHT PASSED')


@contextlib.contextmanager
def lock():
    import msvcrt
    with (OUT/'run.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['draft','preflight','freeze','prepare','train','evaluate','gpu-preflight'])
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--fold',type=int,choices=range(5))
    ap.add_argument('--setting',choices=SETTINGS);ap.add_argument('--seed',type=int,choices=SEEDS);args=ap.parse_args();runtime(args.device)
    with lock():
        if args.mode=='draft':draft()
        elif args.mode=='preflight':preflight()
        elif args.mode=='freeze':freeze()
        elif args.mode=='gpu-preflight':gpu_preflight()
        elif args.mode=='prepare':
            d=assert_frozen()
            for f in range(5) if args.fold is None else [args.fold]:cache_fold(d,SPLITS[f])
        else:
            kwargs=dict(folds=None if args.fold is None else [args.fold],settings=None if args.setting is None else [args.setting],seeds=None if args.seed is None else [args.seed])
            (train if args.mode=='train' else evaluate)(**kwargs)


if __name__=='__main__':main()
