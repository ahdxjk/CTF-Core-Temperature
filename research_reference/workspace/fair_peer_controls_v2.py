"""Matched causal network controls with frozen source splits and participant-level reporting."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import gc
import json
import msvcrt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from prepare_multikernel_backbone_v1 import ROOT, RAW, SPLIT_FILE, SPLITS, SEEDS, PANELS as SOURCE_PANELS, save_json, sha
from external_transfer_v1_data import PANELS as TRANSFER_PANELS
from run_external_transfer_v1 import data, indices, mae, npredict, KEYS, read, protocol as transfer_protocol
from run_raw_tcn_all_v1 import TCN, encoding

OUT = ROOT / "outputs/fair_peer_controls_v2"
WEIGHTS = Path("REPLACE_WITH_MODEL_DIR/fair_peer_controls_v2")
OLD = ROOT / "outputs/external_transfer_v1"
PANELS = dict(TRANSFER_PANELS, Neck14=SOURCE_PANELS["Neck14"])
KINDS = ("CNN", "LSTM", "GRU", "TCN")
ROLE = ("validation", "source_test", "source_thin5", "external")
EXISTING = ("HE_TCN", "MSCA", "BlendFixed", "BlendVal")
OLD_PROTOCOL_HASH = "eb2c3a0434aa24885544bf29efe2ee5c5d9714cb45363128c49c87ac09f3cf41"


def runtime():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "4"
    process = psutil.Process()
    assert {0, 1, 2, 3}.issubset(process.cpu_affinity())
    process.cpu_affinity([0, 1, 2, 3])
    threadpool_limits(4)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.65)
    for p in [OUT, OUT/"records", OUT/"predictions", WEIGHTS]:
        p.mkdir(parents=True, exist_ok=True)


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


def model(kind, channels):
    if kind == "TCN":
        return TCN(channels * 4, 64)
    if kind == "CNN":
        return PlainCNN(channels * 4)
    return Recurrent(kind, channels * 4)


def verify_files(record):
    assert record["status"] == "passed"
    for path, digest in record["hashes"].items():
        assert sha(path) == digest, path


def protocol():
    assert sha(OLD/"protocol.json") == OLD_PROTOCOL_HASH
    assert read(OLD/"protocol.json") == transfer_protocol()
    files = [Path(__file__), RAW, SPLIT_FILE, OLD/"protocol.json", ROOT/"prepare_multikernel_backbone_v1.py"]
    files.extend(ROOT/name for name in read(OLD/"protocol.json")["hashes"])
    old_records = []
    for panel in TRANSFER_PANELS:
        for s in SPLITS:
            base = f"{panel}_fold{s['fold']}_"
            names = [f"TCN_s{seed}" for seed in SEEDS] + list(EXISTING)
            for name in names:
                path = OLD/"records"/(base+name+".json")
                verify_files(read(path))
                old_records.append(path)
    return dict(hashes={str(p):sha(p) for p in files+old_records}, panels=PANELS, splits=SPLITS, seeds=SEEDS,
        task="Current measured rectal temperature. Person-disjoint train24/validation8/test8 x5; no target or phase or ID inputs.",
        input="One-minute whole trials from their start, per-channel train-z last measured value, fresh observation flag, age in hours, ever observed flag; original missingness preserved.",
        architectures={"CNN":"5 plain undilated causal conv layers, kernel29 width24, RF141, per-time LayerNorm, SiLU, dropout .1, residual layers 2-5",
                       "LSTM":"2 unidirectional layers, width64, interlayer dropout .1, linear readout",
                       "GRU":"2 unidirectional layers, width64, interlayer dropout .1, linear readout",
                       "TCN":"source TCN64, eight residual blocks, RF1021"},
        exposure="Identical 8-whole-trial batches, permutations seeded by seed+fold, person-weighted MAE, AdamW lr=.001 wd=.001, clip=1, 150 epochs max, patience20 by validation person MAE. Fixed training budget, differing parameter counts and achieved epochs. Two seeds averaged.",
        thin="Ablate all source test panel sensors at minute mod 5 !=0. Keep original missingness even on grid; retain every original target and one-minute state update. Secondary paired analysis on measured targets at minute mod5=0 only.",
        transfer="ThighHR2/Regional5 only. Mean two seeds then five source fits for same 9 external people/180 observed labels. Never use external labels for selection or scaling. Regional matching and passive immersion.",
        inference="MAE macro across people; 5000 resampled people, paired percentile 95% intervals, fixed seed 20261105, nominal with no multiplicity adjustment. Repeatedly explored source splits. Native and thin on same target keys; paired all baselines versus TCN, archive HE_TCN/MSCA/BlendFixed/BlendVal where available.",
        reuse="Legacy TCN accepted only for transfer panels when original protocol digest, all file digests, checkpoint split, scale and panel match; full reloaded validation/source/external predictions agree within 5e-5 C. Neck14 TCN newly trained.",
        runtime="CUDA FP32, TF32 disabled, memory fraction .65, CPU logical cores 0..3, four threads; no hyperparameter search.",
        intended_seed_models=120, intended_new_fits=100, intended_reused_fits=20)


def prepare():
    frozen = protocol()
    path = OUT/"protocol.json"
    if path.exists():
        assert read(path) == frozen
    else:
        save_json(path, frozen)
    d = data()
    assert d.groupby("trial").size().max() == 141
    assert d.groupby("trial").minute.min().eq(0).all()
    torch.manual_seed(61105)
    tests = {}
    for kind in KINDS:
        m = model(kind, 5).cuda().eval()
        x = torch.randn(3, 141, 20, device="cuda")
        with torch.no_grad():
            predicted = m(x)
            changed = x.clone()
            changed[:, 65:] = torch.randn_like(changed[:, 65:]) * 4
            delta = max((predicted[:, :65]-m(changed)[:, :65]).abs().max().item(),
                        (predicted[:, :65]-m(x[:, :65])).abs().max().item())
            assert delta < 5e-5, (kind, delta)
        m.train()
        m(x).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        tests[kind] = {"prefix_max_error":delta, "parameters_Regional5":sum(p.numel() for p in m.parameters())}
    save_json(OUT/"preflight.json", dict(status="passed", protocol_sha=sha(path), causal_tests=tests,
            gpu=torch.cuda.get_device_name(), affinity=psutil.Process().cpu_affinity(), max_trial_length=141))
    print("PREFLIGHT PASSED", tests, flush=True)


def fold_data(d, split, panel):
    cols = PANELS[panel]
    x,y,valid,rows,ids,groups,w,scale,_ = encoding(d, split, cols)
    groups["source_test"] = groups["test"]
    groups["external"] = np.flatnonzero(ids >= 1000)
    thin = d.copy()
    remove = d.Participant.isin(split["test"]) & (d.minute % 5 != 0)
    thin.loc[remove, cols] = np.nan
    xx,_,vv,rr,_,_,_,ss,_ = encoding(thin, split, cols)
    assert ss == scale and np.array_equal(rr,rows) and np.array_equal(vv,valid)
    assert np.array_equal(xx[groups["train"]], x[groups["train"]])
    ii = indices(d,split)
    ii["source_thin5"] = ii["source_test"]
    # No fabricated on-grid observations, including originally missing measurements.
    flat = rows >= 0
    f = xx.reshape(*xx.shape[:2], len(cols), 4)[...,1][flat]
    assert np.array_equal(f, thin.loc[rows[flat], cols].notna().to_numpy())
    if panel == "Neck14":
        del ii["external"]
    return dict(x=torch.tensor(x,device="cuda"), thin=torch.tensor(xx,device="cuda"),
        y=torch.tensor(y,device="cuda"), w=torch.tensor(w,device="cuda"), groups=groups, ii=ii, rows=rows, scale=scale)


def predictions(m, a, role):
    g = "source_test" if role == "source_thin5" else role
    x = a["thin"] if role == "source_thin5" else a["x"]
    return npredict(m, x, a["groups"][g], a["rows"], a["scale"]).loc[a["ii"][role]].to_numpy()


def commit(tag, paths, **details):
    save_json(OUT/"records"/(tag+".json"), dict(status="passed", protocol_sha=sha(OUT/"protocol.json"),
                  hashes={str(p):sha(p) for p in paths}, **details))


def complete(tag):
    p = OUT/"records"/(tag+".json")
    if not p.exists():
        return False
    r = read(p)
    assert r["protocol_sha"] == sha(OUT/"protocol.json")
    verify_files(r)
    return True


def train_seed(d, a, split, panel, kind, seed):
    tag = f"{panel}_fold{split['fold']}_{kind}_s{seed}"
    if complete(tag):
        return
    tick = time.perf_counter()
    torch.manual_seed(seed+split["fold"])
    rng = np.random.default_rng(seed+split["fold"])
    m = model(kind, len(PANELS[panel])).cuda()
    reuse = kind == "TCN" and panel in TRANSFER_PANELS
    params = sum(p.numel() for p in m.parameters())
    print("START", tag, "parameters", params, "reuse", reuse, flush=True)
    roles = [r for r in ROLE if r in a["ii"]]
    if reuse:
        legacy = read(OLD/"records"/(tag+".json"))
        verify_files(legacy)
        path = Path(legacy["model"])
        history = legacy["history"]
        bestepoch = legacy["bestepoch"]
        assert legacy["split"] == split
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.001)
        tr = a["groups"]["train"]
        best = float("inf")
        history = []
        for epoch in range(1,151):
            m.train()
            order = rng.permutation(tr)
            for start in range(0, len(tr), 8):
                b = order[start:start+8]
                opt.zero_grad(set_to_none=True)
                loss = ((m(a["x"][b]) - a["y"][b]).abs()*a["w"][b]).sum()*len(tr)/len(b)
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
                opt.step()
            val = mae(d, a["ii"]["validation"], predictions(m,a,"validation"))
            history.append(val)
            if val < best:
                best, bestepoch = val, epoch
                state = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            if epoch == 1 or epoch % 10 == 0:
                print(tag, "epoch", epoch, "person_MAE_validation", val, flush=True)
            if epoch - bestepoch >= 20:
                break
        m.load_state_dict(state)
        original = {r:predictions(m,a,r) for r in roles}
        path = WEIGHTS/(tag+".pt")
        tmp = path.with_suffix(".pt.tmp")
        torch.save(dict(state=state, channels=PANELS[panel], split=split, scale=a["scale"], kind=kind, seed=seed),tmp)
        tmp.replace(path)
        del opt
    checkpoint = torch.load(path,map_location="cpu",weights_only=False)
    assert checkpoint["split"] == split and checkpoint["scale"] == a["scale"]
    assert checkpoint["channels"] == PANELS[panel] and checkpoint["kind"] == kind
    m = model(kind,len(PANELS[panel])).cuda()
    m.load_state_dict(checkpoint["state"])
    preds = {r:predictions(m,a,r) for r in roles}
    if reuse:
        with np.load(OLD/"records"/(tag+".npz")) as ref:
            original = {r:ref[r].copy() for r in ref.files}
    replay = max(float(np.max(abs(preds[r]-p))) for r,p in original.items())
    assert replay < 5e-5, (tag,replay)
    assert abs(mae(d,a["ii"]["validation"],preds["validation"])-min(history)) < 5e-5
    assert all(np.isfinite(p).all() for p in preds.values())
    cache = OUT/"predictions"/(tag+".npz")
    with cache.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **preds, **{r+"_row":d.source_excel_row.iloc[a["ii"][r]].to_numpy() for r in roles})
    cache.with_suffix(".tmp").replace(cache)
    commit(tag,[path,cache], model=str(path), reused=reuse, seed=seed, split=split, history=history,
           bestepoch=bestepoch, parameters=params, replay_error=replay, elapsed_seconds=time.perf_counter()-tick)
    print("VERIFIED",tag,flush=True)


def write_csv(df, name):
    path = OUT/name
    tmp = path.with_suffix(path.suffix+".tmp")
    df.to_csv(tmp,index=False,compression="gzip" if name.endswith(".gz") else None)
    tmp.replace(path)
    return path


def combine_fold(d,a,split,panel,kind):
    tag = f"{panel}_fold{split['fold']}_{kind}"
    tags = [tag+f"_s{seed}" for seed in SEEDS]
    assert all(complete(t) for t in tags)
    if complete(tag):
        return
    parts = []
    for role in [r for r in ROLE if r in a["ii"]]:
        q = d.iloc[a["ii"][role]][KEYS].copy()
        pp = []
        for t in tags:
            with np.load(OUT/"predictions"/(t+".npz")) as pred:
                assert np.array_equal(q.source_excel_row,pred[role+"_row"])
                pp.append(pred[role].copy())
        q["prediction"] = np.mean(pp,axis=0)
        q["role"] = role
        parts.append(q)
    path = write_csv(pd.concat(parts), f"predictions/{tag}.csv.gz")
    commit(tag, [path]+[OUT/"records"/(t+".json") for t in tags], seeds=SEEDS, split=split)


def interval(values):
    rng = np.random.default_rng(20261105)
    values = np.asarray(values)
    boots = values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
    return dict(estimate=float(values.mean()), lo=float(np.quantile(boots,.025)), hi=float(np.quantile(boots,.975)), people=len(values))


def report():
    all_rows = []
    for panel in PANELS:
        for kind in KINDS:
            parts = []
            for s in SPLITS:
                tag = f"{panel}_fold{s['fold']}_{kind}"
                assert complete(tag) and all(complete(tag+f"_s{seed}") for seed in SEEDS)
                p = pd.read_csv(OUT/"predictions"/(tag+".csv.gz"))
                p["fold"] = s["fold"]
                parts.append(p)
            p = pd.concat(parts)
            sets = [p[p.role.eq(r)].copy() for r in ["source_test","source_thin5"]]
            for q in sets:
                assert len(q) == q.source_excel_row.nunique() == 17162 and q.Participant.nunique() == 40
            pd.testing.assert_frame_equal(sets[0][KEYS].reset_index(drop=True), sets[1][KEYS].reset_index(drop=True))
            for q in sets[:2]:
                sub = q[q.minute.mod(5).eq(0)].copy()
                sub["role"] = sub.role + "_grid"
                sets.append(sub)
            if panel in TRANSFER_PANELS:
                q = p[p.role.eq("external")].copy()
                assert len(q) == 900 and q.Participant.nunique()==9 and q.groupby("source_excel_row").fold.nunique().eq(5).all()
                avg = q.groupby("source_excel_row").prediction.mean()
                q = q[q.fold.eq(0)].copy()
                q["prediction"] = q.source_excel_row.map(avg)
                sets.append(q)
            for q in sets:
                q["panel"],q["method"] = panel,kind
                all_rows.append(q.drop(columns="fold"))
    all_rows = pd.concat(all_rows,ignore_index=True)
    old = pd.read_csv(OLD/"scored_predictions.csv.gz")
    old = old[old.method.isin(EXISTING)].rename(columns={"dataset":"role"})
    # The archived combined file is convenience output: verify every model prediction against signed per-fold CSVs.
    checked = []
    for (panel, kind, role), g in old.groupby(["panel","method","role"]):
        originals = [pd.read_csv(OLD/"predictions"/f"{panel}_fold{s['fold']}_{kind}_{role}.csv.gz") for s in SPLITS]
        ref = pd.concat(originals).groupby("source_excel_row").prediction.mean()
        assert set(g.source_excel_row)==set(ref.index) and np.allclose(g.prediction, g.source_excel_row.map(ref), rtol=0, atol=5e-6)
        base = all_rows[(all_rows.panel==panel)&(all_rows.method=="TCN")&(all_rows.role==role)]
        pd.testing.assert_frame_equal(g[KEYS].sort_values("source_excel_row").reset_index(drop=True),
                                      base[KEYS].sort_values("source_excel_row").reset_index(drop=True),check_dtype=False)
        checked.append(g)
    all_rows = pd.concat([all_rows,*checked],ignore_index=True)
    err = all_rows.prediction - all_rows.Corerectal
    person = (all_rows.assign(ae=err.abs(),se=err.pow(2),bias=err)
              .groupby(["panel","role","method","Participant"],as_index=False)
              .agg(MAE=("ae","mean"),MSE=("se","mean"),bias=("bias","mean"),labels=("ae","size")))
    person["RMSE"] = np.sqrt(person.MSE)
    summary, paired = [], []
    for (panel,role,kind), g in person.groupby(["panel","role","method"]):
        summary.append(dict(panel=panel,role=role,method=kind,metric="MAE",**interval(g.MAE),RMSE=g.RMSE.mean(),bias=g.bias.mean(),labels=int(g.labels.sum())))
    for (panel,role), g in person.groupby(["panel","role"]):
        pivot = g.pivot(index="Participant",columns="method",values="MAE")
        assert pivot.notna().all().all()
        for ref in ["TCN",*EXISTING]:
            if ref in pivot:
                for kind in KINDS:
                    if kind!=ref:
                        paired.append(dict(panel=panel,role=role,method=kind,reference=ref,**interval(pivot[kind]-pivot[ref])))
    for (panel,kind), g in person[person.method.isin(KINDS)].groupby(["panel","method"]):
        pivot = g.pivot(index="Participant",columns="role",values="MAE")
        for suffix in ["","_grid"]:
            paired.append(dict(panel=panel,role="ablation"+suffix,method=kind+"_thin5",reference=kind+"_native",
                               **interval(pivot["source_thin5"+suffix]-pivot["source_test"+suffix])))
    paths = [write_csv(all_rows,"scored_predictions.csv.gz"),write_csv(person,"person.csv"),
             write_csv(pd.DataFrame(summary),"summary.csv"),write_csv(pd.DataFrame(paired),"paired_ci.csv")]
    save_json(OUT/"completion.json",dict(status="complete",seed_models=120,new_fits=100,reused=20,fold_means=60,
              protocol_sha=sha(OUT/"protocol.json"), hashes={str(p):sha(p) for p in paths}))
    print("COMPLETE verified 120 seed models and paired intervals",flush=True)


def run():
    import datetime
    with (OUT/"training.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        save_json(OUT/"worker.json",dict(pid=os.getpid(), started=datetime.datetime.now().isoformat(),affinity=psutil.Process().cpu_affinity()))
        assert read(OUT/"protocol.json")==protocol()
        assert read(OUT/"preflight.json")["protocol_sha"]==sha(OUT/"protocol.json")
        d = data()
        for split in SPLITS:
            for panel in PANELS:
                a = fold_data(d,split,panel)
                for kind in KINDS:
                    for seed in SEEDS:
                        train_seed(d,a,split,panel,kind,seed)
                    combine_fold(d,a,split,panel,kind)
                del a
                gc.collect()
                torch.cuda.empty_cache()
        report()


if __name__ == "__main__":
    runtime()
    {"prepare":prepare,"run":run}[sys.argv[1]]()

