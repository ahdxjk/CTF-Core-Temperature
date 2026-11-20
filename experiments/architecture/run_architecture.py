"""Portable strict-clean architectural ablations; no imports from the old workspace."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from architecture_core import TCN, ExpandedTCN, encoding, metric, trial_features
from architecture_msca import build_model, check_structure

HERE = Path(__file__).resolve().parent
CLEAN_SHA = "0e6a48308ac45f4fa510a7e9c254a943648488811bf739abbd6619af1edd5495"
BASE = ['abdomen','calf','chest','foot','hand','head','lower arm','lower back','thigh','upper arm','upper back','HR']
CHANNELS = BASE + ['Ins skin','Tamb','BR','Tfront','Rhfront','Tback','Rhback','Temp','Humidity','Wind']
PANELS = {'HeadHR2':['head','HR'], 'Neck2':['Ins skin','Tamb'], 'ESWA4':['head','HR','Temp','Humidity'],
          'Base12':BASE, 'Neck14':BASE+['Ins skin','Tamb'], 'NeckBR15':CHANNELS[:15], 'All22':CHANNELS}
SEEDS = [20261103, 20261104]
MODELS = ['K3', 'MK357', 'State4', 'HE400']
EPOCHS, PATIENCE, BATCH = 150, 20, 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def torch_save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


def log(*args): print(time.strftime('%Y-%m-%d %H:%M:%S'), *args, flush=True)


def pmae(frame):
    return float((frame.prediction-frame.Corerectal).abs().groupby(frame.Participant).mean().mean())


def phase_paired_ci(phase):
    """Pair participant-level phase errors, never individual repeated minutes."""
    rows=[];rng=np.random.default_rng(20261107)
    for (panel,phase_name),g in phase.groupby(['panel','phase_5'],sort=True):
        z=g.pivot(index='Participant',columns='model',values='MAE')
        for method,reference in [('MK357','K3'),('HE400','State4')]:
            if method not in z or reference not in z:continue
            pairs=z[[method,reference]].dropna()
            if pairs.empty:continue
            diff=(pairs[method]-pairs[reference]).to_numpy()
            draws=rng.integers(len(diff),size=(10000,len(diff)))
            lo,hi=np.quantile(diff[draws].mean(1),[.025,.975])
            rows.append({'panel':panel,'phase':phase_name,'method':method,'reference':reference,
                         'method_MAE':float(pairs[method].mean()),'reference_MAE':float(pairs[reference].mean()),
                         'delta_MAE':float(diff.mean()),'CI_low':float(lo),'CI_high':float(hi),
                         'people':len(diff),'improved_people':int((diff<0).sum()),'bootstrap_replicates':10000,
                         'bootstrap_seed':20261107})
    return pd.DataFrame(rows,columns=['panel','phase','method','reference','method_MAE','reference_MAE',
                                     'delta_MAE','CI_low','CI_high','people','improved_people',
                                     'bootstrap_replicates','bootstrap_seed'])


def runtime(args):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the CUDA-enabled PyTorch build; use --device cpu for preflight/smoke.')
    return {'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__, 'pandas':pd.__version__,
            'system':platform.platform(), 'device':args.device, 'threads':args.threads,
            'GPU':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'TF32':False, 'precision':'float32', 'autocast':False}


def load_data(args):
    if sha(args.data) != CLEAN_SHA: raise ValueError('Data hash mismatch: only the verified formula-masked strict-clean data are accepted.')
    if sha(args.splits) != read(HERE/'data_contract.json')['splits_sha256']:
        raise ValueError('Source participant splits changed.')
    d = pd.read_pickle(args.data); splits = read(args.splits)
    assert d.index.equals(pd.RangeIndex(17264)) and d.source_excel_row.is_unique
    assert int(d.Corerectal.notna().sum()) == 17153
    assert d.Participant.nunique() == 40 and d.trial.nunique() == 155
    assert d.groupby('trial').Participant.nunique().eq(1).all()
    assert d.groupby('trial').minute.diff().dropna().eq(1).all()
    assert sorted(p for s in splits for p in s['test']) == sorted(d.Participant.unique())
    for s in splits:
        assert [len(s[k]) for k in ('train','validation','test')] == [24,8,8]
        assert len(set(s['train']+s['validation']+s['test'])) == 40
    return d, splits


def protocol(args, splits):
    return {'version':1, 'data_sha256':sha(args.data), 'splits_sha256':sha(args.splits), 'splits':splits,
            'code_sha256':{p:sha(HERE/p) for p in ('run_architecture.py','architecture_core.py','architecture_msca.py')},
            'channels':CHANNELS, 'panels':PANELS, 'panel_order':list(PANELS), 'seeds':SEEDS,
            'training':{'epochs':150,'patience':20,'lr':.001,'weight_decay':.001,'batch_trials':8,'gradient_clip':1.,
                        'optimizer':'AdamW','loss':'participant-balanced absolute error','target':'current Corerectal',
                        'precision':'float32, TF32 off, no AMP','seed':'seed + source fold'},
            'MSCA':'Shared7, exactly one panel per trial/epoch; balanced seven-epoch cycles; validation mean of seven participant-macro MAEs.',
            'HE':'NeckBR15 only; State4 is width64 TCN, HE400 is the same TCN plus zero-initialized Direct400 history projection. Validation participant-macro MAE.',
            'evaluation':'Five disjoint held-out participant folds; two seeds averaged at prediction level before scoring. No test-based selection.',
            'limitations':'Source splits have been explored before; this is a strict-clean rerun, not a new untouched confirmatory cohort.'}


def freeze(args, splits):
    p = protocol(args, splits); path = args.output/'protocol.json'
    if path.exists(): assert read(path) == p, 'Frozen protocol or code changed; use a separate output directory.'
    else: save(path, p)
    return sha(path)


def build_inputs(obs, panel_indices):
    enabled = torch.tensor([[c in cols for c in CHANNELS] for cols in PANELS.values()], device=obs.device)[panel_indices]
    enabled = enabled.unsqueeze(1).expand(-1, obs.shape[1], -1)
    measured = enabled & obs[...,1].bool()
    return torch.stack((torch.where(measured, obs[...,0], 0.), measured.to(obs.dtype), enabled.to(obs.dtype)), dim=-1)


def trial_schedule(trials, seed):
    rng = np.random.RandomState(seed)
    for epoch in range(EPOCHS):
        if epoch % len(PANELS) == 0:
            offsets = rng.permutation(len(trials)) % len(PANELS)
            panel_order = rng.permutation(len(PANELS))
        order = rng.permutation(len(trials))
        panels = panel_order[(offsets[order]+epoch % len(PANELS)) % len(PANELS)]
        yield trials[order], panels


def preflight(args, d, splits, rt):
    results = {}
    for split in splits:
        x,y,v,rows,ids,groups,w,scale,_ = encoding(d,split,CHANNELS)
        assert int(v.sum()) == 17153 and np.all(w[~v] == 0) and np.isclose(w.sum(),1.)
        train = d.Participant.isin(split['train'])
        np.testing.assert_allclose(scale['mean'],d.loc[train,CHANNELS].mean().fillna(0))
        np.testing.assert_allclose(scale['std'],d.loc[train,CHANNELS].std().fillna(1).clip(lower=.1))
        assert np.isclose(scale['ymean'],d.loc[train,'Corerectal'].mean())
    x,y,v,rows,ids,groups,w,scale,_ = encoding(d,splits[0],CHANNELS)
    slots = x.reshape(*x.shape[:2],22,4)
    obs = torch.tensor(np.stack((np.where(slots[...,1]>0,slots[...,0],0),slots[...,1]),-1)[:7], device=args.device)
    sample = build_inputs(obs,torch.arange(7,device=args.device))
    for j,cols in enumerate(PANELS.values()):
        absent = torch.tensor([c not in cols for c in CHANNELS],device=args.device)
        changed = obs.clone(); changed[j,:,absent] = 123.
        assert torch.equal(sample,build_inputs(changed,torch.arange(7,device=args.device)))
    for name in ('K3','MK357'):
        torch.manual_seed(SEEDS[0]); net = build_model(name).to(args.device).eval()
        with torch.no_grad():
            pred=net(sample); changed=sample.clone(); changed[:,17:]=123.
            future=float((net(changed)[:,:17]-pred[:,:17]).abs().max())
            prefix=float((net(sample[:,:17])-pred[:,:17]).abs().max())
        assert max(future,prefix)<5e-5
        results[name]={**check_structure(net),'future_error':future,'prefix_error':prefix}
    enc = slots[0,:40,:15]; features=trial_features(enc)
    np.testing.assert_allclose(features[:17],trial_features(enc[:17]),atol=1e-5)
    torch.manual_seed(SEEDS[0]); state=TCN(15*4,64).to(args.device).eval()
    torch.manual_seed(SEEDS[0]); he=ExpandedTCN(15,'Direct400').to(args.device).eval()
    a=torch.tensor(features[None],device=args.device); b=a[...,:4].flatten(2)
    with torch.no_grad():
        init=float((state(b)-he(a)).abs().max()); assert init<1e-6
        for name,net,z in (('State4',state,b),('HE400',he,a)):
            pred=net(z); changed=z.clone();changed[:,17:]=123.
            future=float((net(changed)[:,:17]-pred[:,:17]).abs().max())
            prefix=float((net(z[:,:17])-pred[:,:17]).abs().max())
            assert max(future,prefix)<5e-5
            results[name]={'parameters':sum(p.numel() for p in net.parameters()),'future_error':future,'prefix_error':prefix}
    psha=freeze(args,splits)
    result={'status':'passed','protocol_sha256':psha,'rows':len(d),'labels':17153,'people':40,'trials':155,
            'normalization':'training participants only, checked all five folds','target_as_input':False,
            'hidden_sensor_perturbation':'passed','HE_State4_initial_prediction_error':init,'networks':results,'runtime':rt}
    save(args.output/'preflight.json',result); log('PREFLIGHT PASSED',results)


def smoke(args,d,splits,rt):
    preflight(args,d,splits,rt)
    x,y,v,rows,ids,groups,w,scale,_=encoding(d,splits[0],CHANNELS)
    slots=x.reshape(*x.shape[:2],22,4); ix=groups['train'][:2]
    seq=slots[ix,:24]; obs=torch.tensor(np.stack((np.where(seq[...,1]>0,seq[...,0],0),seq[...,1]),-1),device=args.device)
    sample=build_inputs(obs,torch.tensor([0,6],device=args.device))
    f=np.stack([trial_features(s[:,:15]) for s in seq]); expanded=torch.tensor(f,device=args.device)
    target=torch.tensor(y[ix,:24],device=args.device); weight=torch.tensor(w[ix,:24],device=args.device)
    results={}
    for name in MODELS:
        torch.manual_seed(SEEDS[0])
        if name in ('K3','MK357'): net=build_model(name); z=sample
        elif name=='State4':net=TCN(15*4,64);z=expanded[...,:4].flatten(2)
        else:net=ExpandedTCN(15,'Direct400');z=expanded
        net=net.to(args.device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        net.train();opt.zero_grad(set_to_none=True);loss=((net(z)-target).abs()*weight).sum()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():pred=net(z).cpu()
        path=args.output/'smoke'/f'{name}.pt';torch_save(path,{'state':net.state_dict()})
        net.load_state_dict(torch.load(path,map_location=args.device,weights_only=False)['state'])
        with torch.no_grad():error=float((net(z).cpu()-pred).abs().max())
        assert error<5e-5; results[name]={'loss':float(loss.detach()),'checkpoint_replay_error':error}
    save(args.output/'smoke.json',{'status':'passed','purpose':'One training step only; NOT an experiment result.','models':results,'runtime':rt})
    log('CPU/GPU SMOKE PASSED',results)


@contextlib.contextmanager
def output_lock(output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def tag_for(name,seed,fold):
    return f'{name}_{"Shared7" if name in ("K3","MK357") else "NeckBR15"}_s{seed}_f{fold}'


def done(args,tag,psha):
    path=args.output/'records'/f'{tag}.json'
    if not path.exists():return False
    rec=read(path);assert rec['protocol_sha256']==psha
    for rel,h in rec['output_sha256'].items():assert sha(args.output/rel)==h,rel
    assert sha(args.model_dir/rec['checkpoint'])==rec['checkpoint_sha256']
    return True


def train_one(args,d,split,name,seed,psha,rt):
    tag=tag_for(name,seed,split['fold'])
    if done(args,tag,psha):log('VERIFIED; SKIP',tag);return
    started=time.time();log('START',tag)
    shared=name in ('K3','MK357');cols=CHANNELS if shared else PANELS['NeckBR15']
    x,y,v,rows,ids,groups,w,scale,_=encoding(d,split,cols)
    torch.manual_seed(seed+split['fold']);rng=np.random.RandomState(seed+split['fold'])
    if shared:
        slots=x.reshape(*x.shape[:2],22,4)
        obs=torch.tensor(np.stack((np.where(slots[...,1]>0,slots[...,0],0),slots[...,1]),-1),device=args.device)
        views=torch.stack([build_inputs(obs,torch.full((len(obs),),j,device=args.device,dtype=torch.long)) for j in range(7)])
        net=build_model(name).to(args.device);del obs
    elif name=='State4':views=torch.tensor(x,device=args.device);net=TCN(15*4,64).to(args.device)
    else:
        f=np.zeros((*x.shape[:2],15,400),np.float32)
        for i in range(len(x)):
            n=int((rows[i]>=0).sum());f[i,:n]=trial_features(x[i,:n].reshape(n,15,4))
        views=torch.tensor(f,device=args.device);net=ExpandedTCN(15,'Direct400').to(args.device);del f
    ty,tw=torch.tensor(y,device=args.device),torch.tensor(w,device=args.device)
    def predict(role):
        ix=groups[role];net.eval();frames=[]
        with torch.no_grad():
            for j,panel in enumerate(list(PANELS) if shared else ['NeckBR15']):
                xx=views[j] if shared else views
                pred=torch.cat([net(xx[b]) for b in np.array_split(ix,max(1,int(np.ceil(len(ix)/BATCH))))]).cpu().numpy()*scale['ystd']+scale['ymean']
                _,q=metric(d,rows[ix],pred);q['panel']=panel;frames.append(q)
        return pd.concat(frames,ignore_index=True)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best=float('inf');bestepoch=-1;history=[];start_epoch=0;beststate=None
    resume=args.model_dir/'resume'/f'{tag}.pt'
    if resume.exists():
        r=torch.load(resume,map_location='cpu',weights_only=False)
        assert r['protocol_sha256']==psha
        net.load_state_dict(r['state']);opt.load_state_dict(r['optimizer'])
        best,bestepoch,history,beststate=r['best'],r['bestepoch'],r['history'],r['beststate']
        start_epoch=r['next_epoch'];torch.set_rng_state(r['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state(r['cuda_rng'],device=args.device)
        log('RESUME',tag,'next epoch',start_epoch+1)
    schedule=trial_schedule(np.asarray(groups['train']),seed+split['fold']) if shared else ((rng.permutation(groups['train']),None) for _ in range(EPOCHS))
    for epoch,(order,pids) in enumerate(schedule):
        if epoch<start_epoch:continue
        if epoch-bestepoch>PATIENCE:break
        net.train()
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];xx=views[pids[start:start+BATCH],ix] if shared else views[ix]
            opt.zero_grad(set_to_none=True);loss=((net(xx)-ty[ix]).abs()*tw[ix]).sum()*len(order)/len(ix)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        val=predict('validation');score=float(np.mean([pmae(q) for _,q in val.groupby('panel')]))
        history.append(score)
        if score<best:best=score;bestepoch=epoch;beststate={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
        torch_save(resume,{'protocol_sha256':psha,'state':net.state_dict(),'optimizer':opt.state_dict(),'best':best,
                           'bestepoch':bestepoch,'beststate':beststate,'history':history,'next_epoch':epoch+1,
                           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None})
        save(args.output/'status.json',{'status':'running','tag':tag,'epoch':epoch+1,'best_epoch':bestepoch+1,
                                      'validation_MAE':score,'best_validation_MAE':best,'elapsed_seconds':time.time()-started})
        if epoch==0 or (epoch+1)%10==0:log(tag,'epoch',epoch+1,'validation MAE',round(score,6))
        if epoch-bestepoch>=PATIENCE:break
    net.load_state_dict(beststate)
    frames={role:predict(role) for role in ('validation','test')}
    checkpoint=args.model_dir/f'{tag}.pt'
    torch_save(checkpoint,{'state':beststate,'scale':scale,'split':split,'model':name,'seed':seed,'protocol_sha256':psha})
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state'])
    replay=max(float(np.max(np.abs(predict(role).prediction-q.prediction))) for role,q in frames.items())
    assert replay<5e-5
    hashes={}
    for role,q in frames.items():
        path=args.output/'predictions'/f'{tag}_{role}.csv.gz';path.parent.mkdir(parents=True,exist_ok=True)
        q.to_csv(path,index=False);hashes[path.relative_to(args.output).as_posix()]=sha(path)
    save(args.output/'records'/f'{tag}.json',{'status':'passed','protocol_sha256':psha,'model':name,'seed':seed,'split':split,
         'checkpoint':checkpoint.name,'checkpoint_sha256':sha(checkpoint),'output_sha256':hashes,'best_epoch':bestepoch+1,
         'history':history,'checkpoint_replay_error':replay,'seconds_this_invocation':time.time()-started,'runtime':rt})
    log('DONE',tag,'best epoch',bestepoch+1,'validation MAE',round(best,6))
    del net,opt,views,ty,tw;gc.collect()
    if args.device.startswith('cuda'):torch.cuda.empty_cache()


def report(args,d,splits,psha):
    combined=[];complete=[];missing=[]
    for name in MODELS:
        tags=[tag_for(name,seed,s['fold']) for s in splits for seed in SEEDS]
        absent=[tag for tag in tags if not done(args,tag,psha)]
        if absent:missing.extend(absent);continue
        parts=[]
        for split in splits:
            seeds=[]
            for seed in SEEDS:
                q=pd.read_csv(args.output/'predictions'/f'{tag_for(name,seed,split["fold"])}_test.csv.gz')
                assert set(q.Participant)==set(split['test'])
                q=q.set_index(['panel','source_excel_row']).sort_index();seeds.append(q)
            assert seeds[0].index.equals(seeds[1].index)
            np.testing.assert_allclose(seeds[0].Corerectal,seeds[1].Corerectal)
            q=seeds[0].copy();q['prediction']=np.mean([z.prediction.to_numpy() for z in seeds],axis=0)
            q['fold']=split['fold'];parts.append(q.reset_index())
        q=pd.concat(parts,ignore_index=True);q['model']=name
        assert all(len(z)==17153 for _,z in q.groupby('panel'))
        assert not q.duplicated(['panel','source_excel_row']).any()
        combined.append(q);complete.append(name)
    save(args.output/'completion.json',{'status':'complete' if not missing else 'partial','complete_models':complete,
                                       'missing_fits':missing,'fits_completed':40-len(missing),'fits_total':40})
    if not combined:log('No complete two-seed, five-fold model yet.');return
    q=pd.concat(combined,ignore_index=True);q.to_csv(args.output/'oof_predictions.csv.gz',index=False)
    person=q.assign(AE=(q.prediction-q.Corerectal).abs()).groupby(['panel','model','Participant']).AE.mean().reset_index(name='MAE')
    person.to_csv(args.output/'participant_mae.csv',index=False)
    summary=person.groupby(['panel','model']).MAE.mean().reset_index();summary.to_csv(args.output/'summary.csv',index=False)
    phase=q.assign(AE=(q.prediction-q.Corerectal).abs()).groupby(['panel','model','phase_5','Participant']).AE.mean().reset_index(name='MAE')
    phase.to_csv(args.output/'phase_participant_mae.csv',index=False)
    phase_paired_ci(phase).to_csv(args.output/'phase_paired_ci.csv',index=False)
    comparisons=[];rng=np.random.default_rng(20261107)
    for panel,g in person.groupby('panel'):
        z=g.pivot(index='Participant',columns='model',values='MAE')
        for method,reference in [('MK357','K3'),('HE400','State4')]:
            if method not in z or reference not in z:continue
            diff=(z[method]-z[reference]).to_numpy();draws=rng.integers(len(diff),size=(10000,len(diff)))
            lo,hi=np.quantile(diff[draws].mean(1),[.025,.975])
            comparisons.append({'panel':panel,'method':method,'reference':reference,'delta_MAE':float(diff.mean()),
                                'CI_low':float(lo),'CI_high':float(hi),'people':len(diff),'improved_people':int((diff<0).sum())})
    pd.DataFrame(comparisons).to_csv(args.output/'paired_ci.csv',index=False)
    log('REPORT',summary.to_dict('records'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','smoke','train','report','status'])
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('STRICT_ARCH_DATA',HERE/'data/clean_raw.pkl')))
    parser.add_argument('--splits',type=Path,default=HERE/'data/splits.json')
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('STRICT_ARCH_OUTPUT',HERE/'results')))
    parser.add_argument('--model-dir',type=Path,default=None)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--folds',nargs='+',type=int,choices=range(5),default=list(range(5)))
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=SEEDS)
    args=parser.parse_args();args.output=args.output.resolve();args.model_dir=(args.model_dir or args.output/'checkpoints').resolve()
    args.output.mkdir(parents=True,exist_ok=True);args.model_dir.mkdir(parents=True,exist_ok=True)
    rt=runtime(args);d,splits=load_data(args)
    if args.mode=='status':
        for name in ('status.json','completion.json'):
            if (args.output/name).exists():print(json.dumps(read(args.output/name),indent=2))
        return
    if args.mode in ('preflight','smoke'):
        (preflight if args.mode=='preflight' else smoke)(args,d,splits,rt);return
    psha=freeze(args,splits)
    if args.mode=='report':report(args,d,splits,psha);return
    with output_lock(args.output):
        preflight(args,d,splits,rt)
        save(args.output/'runtime.json',rt)
        for name in args.models:
            for s in splits:
                if s['fold'] not in args.folds:continue
                for seed in args.seeds:train_one(args,d,s,name,seed,psha,rt)
            report(args,d,splits,psha)
        report(args,d,splits,psha)
        save(args.output/'status.json',{'status':'requested_jobs_complete','models':args.models,'folds':args.folds,'seeds':args.seeds})


if __name__=='__main__':main()
