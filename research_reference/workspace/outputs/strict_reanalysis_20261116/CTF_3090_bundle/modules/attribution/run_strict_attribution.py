"""Strict-data rerun of the legacy All22 CTF attribution study.

Usage: --stage preflight | shap | neural | fullblock | summarize.
Existing legacy results are never overwritten or used as numerical results.
Frozen strictly refitted models are explained, with exact archive replay gates.
"""
from pathlib import Path
import os, sys, json, hashlib, argparse, ast, time, types, math, gc
_early=argparse.ArgumentParser(add_help=False)
_early.add_argument('--root'); _early.add_argument('--output')
_paths,_unknown=_early.parse_known_args()
_candidate=_paths.root or os.environ.get('STRICT_RESEARCH_ROOT')
ROOT=Path(_candidate).resolve() if _candidate else next(p for p in Path(__file__).resolve().parents if (p/'multikernel_backbone_model_v1.py').exists())
BASE=Path(_paths.output).resolve() if _paths.output else ROOT/'outputs/strict_reanalysis_attribution'
BASE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = 'true'
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import joblib
from scipy.stats import spearmanr
from multikernel_backbone_model_v1 import build_model

def source_definitions(filename, namespace, names=None):
    """Import audited definitions only: historical launch-time drive paths never execute."""
    path=ROOT/filename
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
    if names is not None: assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

_network={'torch':torch,'nn':torch.nn,'tf':torch.nn.functional}
source_definitions('run_raw_tcn_all_v1.py',_network,['Block','TCN'])
_network.update(F=torch.nn.functional,MODELS=('Shared400','Direct400'))
source_definitions('channel_bank400_v1.py',_network,['ExpandedTCN'])
_bank={'np':np,'pd':pd,'WINDOWS':(3,5,7,10,12,15,20,25,30,45,60,90)}
source_definitions('channel_bank400_v1.py',_bank,['trial_features'])

_groups=json.loads((ROOT/'outputs/interpretability_v1/protocol.json').read_text(encoding='utf8'))['groups']
_channels=[c for group in _groups.values() for c in group]
_coalitions=((np.arange(2**len(_groups))[:,None]>>np.arange(len(_groups)))&1).astype(bool)
_ch_group=np.array([next(i for i,g in enumerate(_groups.values()) if c in g) for c in _channels])
_splits=json.loads((ROOT/'outputs/cpu_baselines_v1/splits.json').read_text())
shap=types.ModuleType('strict_shap_functions')
shap.__dict__.update(ROOT=ROOT,BASE=BASE,OUT=BASE/'shap',Path=Path,np=np,pd=pd,torch=torch,joblib=joblib,gc=gc,math=math,time=time,hashlib=hashlib,json=json,
    GROUPS=_groups,GROUP_NAMES=list(_groups),CHANNELS=_channels,G=len(_groups),COALITIONS=_coalitions,CH_GROUP=_ch_group,CH_COAL=_coalitions[:,_ch_group],
    SEEDS=[20261103,20261104],MEMBERS=['TabPFN','MSCA','HE-TCN','ExtraTrees'],SPLITS=_splits,build_model=build_model,ExpandedTCN=_network['ExpandedTCN'],trial_features=_bank['trial_features'],__file__=str(ROOT/'xai_shap_v1.py'))
source_definitions('xai_shap_v1.py',shap.__dict__)
neural=types.ModuleType('strict_neural_functions')
neural.__dict__.update(ROOT=ROOT,OUT=BASE/'neural',SHARED=BASE,Path=Path,np=np,pd=pd,torch=torch,hashlib=hashlib,json=json,os=os,time=time,spearmanr=spearmanr,
    CHANNELS=_channels,SEEDS=[20261103,20261104],SPLITS=_splits,build_model=build_model,RNG_SEED=20261106,
    LAG_BINS=((0,4,'0-4'),(5,14,'5-14'),(15,29,'15-29'),(30,59,'30-59'),(60,10000,'60+')),DEVICE=torch.device('cpu'),__file__=str(ROOT/'xai_neural_v1.py'))
source_definitions('xai_neural_v1.py',neural.__dict__)

SOURCE = ROOT/'outputs/strict_confirmation_v1'
COMPLETION = ROOT/'outputs/ctf_full_coverage_20261115/base_completion'
SEEDS = [20261103, 20261104]
SPLITS = json.loads((ROOT/'outputs/cpu_baselines_v1/splits.json').read_text())
CHANNELS = shap.CHANNELS
DEVICE = 'cpu'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def clean_load():
    path=SOURCE/'clean_raw.pkl'
    assert sha(path)=='0e6a48308ac45f4fa510a7e9c254a943648488811bf739abbd6619af1edd5495'
    return pd.read_pickle(path)

def resolve_original_path(path):
    """Resolve archived absolute paths against the portable root without editing archives."""
    raw=str(path).replace('\\','/')
    if '/outputs/' in raw:
        candidate=ROOT/'outputs'/raw.split('/outputs/',1)[1]
        if candidate.exists(): return candidate
    # Model suffixes are stable and retain full subdirectory identity.
    for suffix in ['ctf_full_coverage_20261115/base_completion/','strict_confirmation_v1/']:
        if suffix in raw:
            candidate=ROOT/'checkpoints'/suffix/raw.split(suffix,1)[1]
            if candidate.exists(): return candidate
    raise FileNotFoundError(f'Bundle is missing archived asset: {path}')

def normalize_record(rec):
    rec=dict(rec)
    rec['original_model_path']=rec['model']
    rec['model']=str(resolve_original_path(rec['model']))
    rec['hashes']={str(resolve_original_path(path)):digest for path,digest in rec['hashes'].items()}
    return rec
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def record_path(fold, family, seed=None):
    if family=='MSCA': return SOURCE/'records'/f'MSCA_Shared7_s{seed}_f{fold}.json'
    if family=='HE-TCN': return COMPLETION/'records'/f'HE_All22_s{seed}_f{fold}.json'
    return COMPLETION/'records'/f'{"PFN" if family=="TabPFN" else "ET"}_All22_f{fold}.json'

def records(fold,family):
    return [normalize_record(read(record_path(fold,family,seed))) for seed in (SEEDS if family in ('MSCA','HE-TCN') else [None])]

def prediction_path(record, role='test'):
    paths=[Path(p) for p in record['hashes'] if p.endswith(f'_{role}.csv.gz')]
    assert len(paths)==1
    return paths[0]

def archive_one(record):
    a=pd.read_csv(prediction_path(record))
    if 'panel' in a: a=a.loc[a.panel.eq('All22')]
    return a.set_index('source_excel_row').prediction

def archived(fold,family):
    arrays=[archive_one(rec) for rec in records(fold,family)]
    for a in arrays[1:]: assert a.index.equals(arrays[0].index)
    return pd.concat(arrays,axis=1).mean(axis=1)

def load_neural(fold,family,device=None):
    device=DEVICE if device is None else device
    models=[]; scale=None; paths=[]
    for rec in records(fold,family):
        path=Path(rec['model']); assert sha(path)==rec['hashes'][str(path)]
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['split']==SPLITS[fold]
        s=ck['scale']; assert s['columns']==CHANNELS
        if scale is not None: assert s==scale
        scale=s
        net=shap.build_model('MK357') if family=='MSCA' else shap.ExpandedTCN(22,'Direct400')
        net.load_state_dict(ck['state']); net.to(device).eval()
        for p in net.parameters(): p.requires_grad_(False)
        models.append(net); paths.append(str(path))
    return models,scale,paths

def frozen_tab(fold,family):
    rec=records(fold,family)[0]; path=Path(rec['model'])
    assert sha(path)==rec['hashes'][str(path)]
    if family=='TabPFN':
        from tabpfn_memory_runtime_v1 import configure_pfn_chunks
        configure_pfn_chunks(65536)
        model=load_portable_tabpfn(path)
        model.n_preprocessing_jobs=2
        model.keep_cache_on_device=False
        model.executor_.keep_cache_on_device=False
        model.memory_saving_mode=True
        if DEVICE=='cuda': model.to('cuda')
    else:
        import joblib
        model=joblib.load(path); model.named_steps['model'].n_jobs=2
    return model,dict(rec,model_path=str(path),selected_output=rec['settings']['PFN_output'])

def load_portable_tabpfn(path):
    """Upstream fitted loader, with only its saved absolute base-weight path relocated.

    Fitted parameters, context rows, preprocessor, caches and output type are untouched.
    The original fitted archive remains byte-identical and hash-verified.
    """
    import tempfile
    from tabpfn.model_loading import _extract_archive
    from tabpfn.inference import InferenceEngine
    from tabpfn import TabPFNRegressor
    weights=ROOT/'assets/tabpfn/tabpfn-v3.5-20260909.safetensors'
    if not weights.exists(): weights=ROOT/'external/TabPFN35/tabpfn-v3.5-20260909.safetensors'
    assert weights.exists(),'Exact TabPFN v3.5 base weights are missing'
    assert sha(weights)=='ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3','TabPFN base weight checksum mismatch'
    with tempfile.TemporaryDirectory(dir=BASE/'temp') as directory:
        tmp=Path(directory); _extract_archive(Path(path),tmp)
        params=read(tmp/'init_params.json')
        assert params.pop('__class_name__')=='TabPFNRegressor'
        params['model_path']=str(weights)
        if isinstance(params.get('inference_precision'),str) and params['inference_precision'].startswith('torch.'):
            params['inference_precision']=getattr(torch,params['inference_precision'].split('.')[1])
        params['device']='cpu'; params['n_preprocessing_jobs']=2
        model=TabPFNRegressor(**params); model._initialize_model_variables()
        attrs=joblib.load(tmp/'fitted_attrs.joblib')
        for key,value in attrs.items(): setattr(model,key,value)
        model.executor_=InferenceEngine.load_state(tmp/'executor_state.joblib',model.models_)
        model.to('cpu')
    return model

class FrozenStrictMSCA(neural.FrozenMSCA):
    def __init__(self,fold):
        self.fold=int(fold)
        self.models,self.scale,paths=load_neural(fold,'MSCA',DEVICE)
        self.hashes={str(seed):sha(path) for seed,path in zip(SEEDS,paths)}

def configure():
    shap.BASE=BASE; shap.OUT=BASE/'shap'; shap.OUT.mkdir(exist_ok=True)
    shap.TMP=BASE/'temp'; shap.TMP.mkdir(exist_ok=True)
    shap.load_neural=load_neural; shap.frozen_tab=frozen_tab; shap.archived=archived
    original_neural_predict=shap.neural_predict
    shap.neural_predict=lambda models,prefix,ref,scale,family,device=None: original_neural_predict(models,prefix,ref,scale,family,DEVICE)
    neural.SHARED=BASE; neural.OUT=BASE/'neural'; neural.DEVICE=torch.device(DEVICE)
    neural.FrozenMSCA=FrozenStrictMSCA; neural.load_data=clean_load
    # Replace only the archived prediction reader and the legacy thread count.
    # All IG, replacement, donor-selection and randomization mathematics stay intact.
    import inspect
    source=inspect.getsource(neural.run)
    old="""        archived = []
        for seed in SEEDS:
            p = ROOT / f'outputs/multikernel_backbone_v1/fits/MK357_s{seed}_f{fold}/test.csv.gz'
            archived.append(pd.read_csv(p).query(\"panel == 'All22'\").set_index('row').pred)
        archived = pd.concat(archived, axis=1)"""
    new="""        archived = pd.concat([strict_archive_one(r) for r in strict_records(fold,'MSCA')],axis=1)"""
    assert source.count(old)==1
    source=source.replace(old,new).replace('torch.set_num_threads(4)','torch.set_num_threads(2)')
    neural.strict_archive_one=archive_one; neural.strict_records=records
    exec(compile(source,str(BASE/'strict_neural_run.py'),'exec'),neural.__dict__)
    (BASE/'strict_neural_run.py').write_text(source,encoding='utf8')
    old_protocol=neural.protocol
    def strict_protocol():
        p=old_protocol()
        p.update(data_sha256=sha(SOURCE/'clean_raw.pkl'),
                 strict_driver_sha256=sha(__file__),
                 strict_preflight_sha256=sha(BASE/'preflight.json'),
                 amendment='Strict formula-masked rerun; identical explanatory mathematics; lower-middle endpoint selection reapplied after removing formula-derived targets. Model and prediction files are strict refits. Existing-fold exploratory evidence, not independent confirmation.')
        return p
    neural.protocol=strict_protocol

def preflight():
    data=clean_load()
    assert len(data)==17264 and data.Corerectal.notna().sum()==17153
    rows=[]
    for split in SPLITS:
        assert not(set(split['train'])&set(split['test']))
        assert not(set(split['validation'])&set(split['test']))
        for person in sorted(split['test']):
            for phase in ['P1','P2','P3','P4','P5']:
                pool=data.loc[data.Participant.eq(person)&data.phase_5.eq(phase)&data.Corerectal.notna()].sort_values(['trial','minute','source_excel_row'])
                assert len(pool)
                row=pool.iloc[(len(pool)-1)//2]
                prefix=data.loc[data.trial.eq(row.trial)&data.minute.le(row.minute)]
                rows.append(dict(sample_id=len(rows),fold=split['fold'],person=person,phase=phase,trial=int(row.trial),minute=int(row.minute),row=int(row.source_excel_row),data_index=int(row.name),y=float(row.Corerectal),prefix_length=len(prefix),eligible=len(pool)))
    selection=pd.DataFrame(rows); assert len(selection)==200 and selection.person.nunique()==40
    selection.to_csv(BASE/'selection.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/interpretability_v1/selection.csv')
    changed=selection.loc[selection.row.ne(old.row),['sample_id','person','phase','row']].copy()
    changed['old_row']=old.loc[changed.index,'row']
    changed.to_csv(BASE/'changed_endpoints.csv',index=False)
    source_records=[]; replay=[]
    for fold in range(5):
        chosen=selection.loc[selection.fold.eq(fold)]
        train=data.loc[data.Participant.isin(SPLITS[fold]['train'])]
        for family in shap.MEMBERS:
            for rec in records(fold,family):
                assert rec['status']=='passed' and rec['split']==SPLITS[fold]
                for path,digest in rec['hashes'].items(): assert sha(path)==digest,(path,'hash mismatch')
                rpath=record_path(fold,family,rec.get('seed'))
                source_records.append(dict(fold=fold,family=family,seed=rec.get('seed'),record=str(rpath),record_sha256=sha(rpath),model=rec['model'],model_sha256=rec['hashes'][rec['model']],test_archive=str(prediction_path(rec)),test_archive_sha256=sha(prediction_path(rec))))
                original=archive_one(rec)
                assert set(chosen.row).issubset(set(original.index))
            if family in ('MSCA','HE-TCN'):
                models,scale,paths=load_neural(fold,family,'cpu')
                np.testing.assert_allclose(scale['mean'],train[CHANNELS].mean().to_numpy(),atol=1e-12)
                np.testing.assert_allclose(scale['std'],train[CHANNELS].std().replace(0,1).fillna(1).to_numpy(),atol=1e-12)
                by_seed=[]
                for row in chosen.itertuples(index=False):
                    prefix=data.loc[data.trial.eq(row.trial)&data.minute.le(row.minute)]
                    x,_=shap.neural_coalitions(prefix,prefix,scale,family)
                    with torch.no_grad():
                        out=np.array([float((m(torch.from_numpy(x[None]))[0,-1]*scale['ystd']+scale['ymean']).item()) for m in models])
                    expected=np.array([archive_one(r).loc[row.row] for r in records(fold,family)])
                    err=float(np.max(np.abs(out-expected)))
                    assert err<5e-5,(family,fold,row.row,err)
                    by_seed.append(err)
                replay.append(dict(fold=fold,family=family,samples=len(chosen),max_error_C=max(by_seed)))
                del models
            elif family=='ExtraTrees':
                model,rec=frozen_tab(fold,family)
                q=pd.concat([shap.tab_features(data.loc[data.trial.eq(row.trial)&data.minute.le(row.minute)]).iloc[[-1]] for row in chosen.itertuples(index=False)],ignore_index=True)
                out=model.predict(q.loc[:,model.feature_names_in_]); expected=archived(fold,family).loc[chosen.row].to_numpy()
                err=float(np.max(np.abs(out-expected))); assert err<5e-5
                replay.append(dict(fold=fold,family=family,samples=len(chosen),max_error_C=err)); del model
        print('PREFLIGHT FOLD',fold,'passed',flush=True)
    pd.DataFrame(source_records).to_csv(BASE/'strict_sources.csv',index=False)
    pd.DataFrame(replay).to_csv(BASE/'preflight_replay.csv',index=False)
    # Strict Neck14 engineering audit is valid separate evidence; check its model identity.
    engineering=[]
    for fold in range(5):
        for family,tag in [('MSCA','MSCA_Shared7'),('HE-TCN','HE_Neck14')]:
            for seed in SEEDS:
                a=read(ROOT/f'outputs/revision_round3_20261108/estimation/records/{tag}_s{seed}_f{fold}.json')
                b=read(SOURCE/f'records/{tag}_s{seed}_f{fold}.json')
                assert a['model']==b['model'] and a['hashes'][a['model']]==b['hashes'][b['model']]
                engineering.append(dict(fold=fold,family=family,seed=seed,identical=True))
    save(BASE/'preflight.json',dict(status='passed',raw_sha256=sha(SOURCE/'clean_raw.pkl'),samples=200,participants=40,phase_counts=selection.phase.value_counts().sort_index().to_dict(),changed_endpoints=len(changed),model_artifacts=len(source_records),replay=replay,PFN_replay='Deferred to each exact-Shapley job; all checkpoint and test/validation archive hashes verified. No PFN result released before endpoint replay gate.',strict_Neck14_engineering_model_identity_count=len(engineering),code_sha256=sha(__file__),source_code_sha256={name:sha(ROOT/name) for name in ['xai_shap_v1.py','xai_neural_v1.py','xai_fullblock_control_v1.py','run_panel_peer_tables_v1.py','channel_bank400_v1.py','multikernel_backbone_model_v1.py']},selection_sha256=sha(BASE/'selection.csv')))
    protocol=read(ROOT/'outputs/interpretability_v1/protocol.json')
    protocol.update(version=2,status='strict_rerun_frozen_before_attribution',files={'raw_sha256':sha(SOURCE/'clean_raw.pkl'),'selection_sha256':sha(BASE/'selection.csv'),'splits_sha256':sha(ROOT/'outputs/cpu_baselines_v1/splits.json')},strict_sources_sha256=sha(BASE/'strict_sources.csv'),strict_preflight_sha256=sha(BASE/'preflight.json'))
    save(BASE/'protocol.json',protocol)
    print('STRICT PREFLIGHT PASSED',flush=True)

def summarize():
    result={'status':'incomplete'}
    selection=pd.read_csv(BASE/'selection.csv')
    if shap.aggregate(selection):
        a=pd.read_csv(BASE/'shap/group_shap.csv')
        person=a.groupby(['model','group','person'],as_index=False)[['phi','absolute_phi']].mean()
        summary=person.groupby(['model','group'],as_index=False)[['phi','absolute_phi']].mean()
        person.to_csv(BASE/'shap/person_summary.csv',index=False)
        summary.to_csv(BASE/'shap/summary.csv',index=False)
        result['shap']='complete'
    neural_completion=read(BASE/'neural/completion.json') if (BASE/'neural/completion.json').exists() else {}
    if neural_completion.get('samples')==200 and neural_completion.get('people')==40 and neural_completion.get('folds')==list(range(5)):
        o=pd.read_csv(BASE/'neural/occlusion.csv')
        donor_ids=set(pd.read_csv(BASE/'neural/donors.csv').sample_id)
        records=[]
        for metric in ['abs_change_C','delta_absolute_error_C']:
            for scope,table in [('all',o),('P3_matched40',o.loc[o.sample_id.isin(donor_ids)])]:
                for (reference,fraction),group in table.groupby(['reference','fraction']):
                    cases=group.groupby(['person','sample_id','order'])[metric].mean().unstack('order')
                    cases['difference']=cases.top-cases.random
                    people=cases.groupby('person').mean()
                    delta=people.difference.to_numpy(); rng=np.random.default_rng(20261106)
                    interval=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975])
                    records.append(dict(metric=metric,scope=scope,reference=reference,fraction=fraction,people=len(people),top=float(people.top.mean()),random=float(people.random.mean()),top_minus_random=float(delta.mean()),ci_low=float(interval[0]),ci_high=float(interval[1])))
        pd.DataFrame(records).to_csv(BASE/'neural/strict_fidelity_summary.csv',index=False)
        r=pd.read_csv(BASE/'neural/randomization.csv')
        r.groupby('mode')[['signed_spearman','absolute_spearman']].mean().to_csv(BASE/'neural/strict_randomization_summary.csv')
        result['neural']='complete'
    if result.get('shap')=='complete' and result.get('neural')=='complete' and (BASE/'neural/fullblock/completion.json').exists(): result['status']='complete'
    save(BASE/'status.json',result)

def main():
    global DEVICE
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['preflight','shap','neural','fullblock','summarize'],required=True)
    p.add_argument('--root'); p.add_argument('--output')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--family',choices=shap.MEMBERS)
    p.add_argument('--fold',type=int); p.add_argument('--benchmark',action='store_true')
    args=p.parse_args(); DEVICE=args.device; configure()
    with threadpool_limits(limits=2):
        if args.stage=='preflight': preflight(); return
        pre=read(BASE/'preflight.json'); assert pre['status']=='passed'
        assert pre['raw_sha256']==sha(SOURCE/'clean_raw.pkl')
        assert pre['selection_sha256']==sha(BASE/'selection.csv')
        if args.stage=='shap':
            data=clean_load(); selection=pd.read_csv(BASE/'selection.csv')
            for fold in ([args.fold] if args.fold is not None else range(5)):
                for family in ([args.family] if args.family else shap.MEMBERS):
                    shap.run_family(fold,family,data,selection,args.benchmark)
            if not args.benchmark: shap.aggregate(selection)
        elif args.stage=='neural':
            neural.run(None if args.fold is None else [args.fold],1 if args.benchmark else None)
            if not args.benchmark:
                report=read(BASE/'neural/completion.json')
                if report['samples']!=200 or report['people']!=40 or report['folds']!=list(range(5)):
                    report['status']='partial'
                    save(BASE/'neural/completion.json',report)
        elif args.stage=='fullblock':
            full=types.ModuleType('strict_fullblock_functions')
            full.__dict__.update(Path=Path,json=json,time=time,np=np,pd=pd,torch=torch,FrozenMSCA=FrozenStrictMSCA,ROOT=ROOT,SHARED=BASE,OUT=BASE/'neural/fullblock',
                SPLITS=SPLITS,RNG_SEED=neural.RNG_SEED,sha=sha,freeze=neural.freeze,dump=neural.dump,flat_reference=neural.flat_reference,metadata=neural.metadata,__file__=str(ROOT/'xai_fullblock_control_v1.py'))
            source_definitions('xai_fullblock_control_v1.py',full.__dict__)
            import inspect
            source=inspect.getsource(full.main).replace('torch.set_num_threads(4)','torch.set_num_threads(2)')
            exec(compile(source,str(BASE/'strict_fullblock_run.py'),'exec'),full.__dict__)
            full.main()
        else: summarize()

if __name__=='__main__': main()
