"""Exploratory post-result control using only complete five-minute blocks.

This supplements, and does not replace, the original frozen explanation study.
It equalizes replaced elapsed minutes at a fixed block budget, while retaining
the original masks (observed sensor-cell counts may still differ).
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch

from xai_neural_v1 import FrozenMSCA, ROOT, SHARED, SPLITS, RNG_SEED, sha, freeze, dump, flat_reference, metadata

OUT = SHARED / 'neural/fullblock'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = pd.read_csv(SHARED/'selection.csv')
    donors = pd.read_csv(SHARED/'neural/donors.csv')
    cells = np.load(SHARED/'neural/cell_attribution.npz')
    protocol = dict(
        scope='Exploratory robustness control added after observing primary results because the oldest partial block can differ in duration. Not a new confirmatory experiment; original outputs unchanged.',
        hypotheses='Whether primary IG-ranked blocks change model predictions more than random-ranked blocks when replaced elapsed duration is exactly matched.',
        selection='Existing 200 predefined person-phase cases for train_mean; existing 40 P3 donor cases; no new sample or residual selection.',
        candidates='Only complete 5-minute blocks counting backwards from target. Floor(prefix_length/5) candidates; earliest partial block is never replaced. Zero candidates retained in eligibility and occlusion tables as not_applicable.',
        ranking='Existing train_mean mean-two-seed IG, sum absolute cell attribution within each complete block. Stable top/least ranks.',
        perturbation='ceil(budget*n_complete_blocks), budgets 10/30/50%; top/least/5 seeded random permutations. Replace values only, retain masks. Same block count gives same duration, not necessarily equal observed sensor-cell count.',
        references='Training-fold mean and unchanged existing training donor input, including already documented one-minute donor padding.',
        uncertainty='Average random repetitions within target, then target differences within participant; 10000 paired participant bootstrap, frozen models conditional uncertainty, unadjusted exploratory intervals.',
        seed=RNG_SEED,
        source_hashes={str(path.relative_to(ROOT)):sha(path) for path in
                       [Path(__file__), ROOT/'xai_neural_v1.py', SHARED/'selection.csv',
                        SHARED/'neural/cell_attribution.npz', SHARED/'neural/donors.csv']})
    freeze(OUT/'protocol.json', protocol)
    donor_ids = set(donors.sample_id)
    records, eligibility = [], []
    started = time.perf_counter()
    for split in SPLITS:
        model = FrozenMSCA(split['fold'])
        for _, row in selection[selection.fold.eq(split['fold'])].iterrows():
            sid = int(row.sample_id)
            x = cells[f's{sid}_input']
            attribution = cells[f's{sid}_train_mean']
            n = len(x)
            n_full = n // 5
            block = (n-1-np.arange(n)) // 5
            references = [('train_mean', flat_reference(x, np.zeros(x.shape[1])))]
            if sid in donor_ids:
                references.append(('training_donor', cells[f's{sid}_donor_input']))
            for reference, ref in references:
                status = 'eligible' if n_full else 'not_applicable'
                base_meta = dict(**metadata(row), reference=reference, rank_source='train_mean',
                                 eligibility=status, full_blocks=n_full, total_blocks=n_full,
                                 ignored_oldest_minutes=n%5, prefix_length=n)
                eligibility.append(base_meta)
                if not n_full:
                    records.append(dict(**base_meta, order='not_applicable'))
                    continue
                scores = np.array([np.abs(attribution[block.eq(b) if hasattr(block, 'eq') else block==b]).sum()
                                   for b in range(n_full)])
                top = np.argsort(-scores, kind='stable')
                least = np.argsort(scores, kind='stable')
                rng = np.random.default_rng(RNG_SEED + sid*73)
                random_orders = [rng.permutation(n_full) for _ in range(5)]
                pred = float(model.predict(x)[0])
                inputs, details = [], []
                for fraction in (.1,.3,.5):
                    k = max(1,int(np.ceil(fraction*n_full)))
                    for order, permutations in [('top',[top]),('least',[least]),('random',random_orders)]:
                        for repeat, permutation in enumerate(permutations):
                            replace = np.isin(block,permutation[:k])
                            assert replace.sum()==5*k
                            assert not np.any(replace & (block >= n_full))
                            modified=x.copy()
                            modified[replace,:,0]=ref[replace,:,0]
                            assert np.array_equal(modified[...,1:],x[...,1:])
                            inputs.append(modified)
                            details.append(dict(**base_meta,order=order,fraction=fraction,repeat=repeat,
                                                blocks=k,actual_fraction=k/n_full,replaced_minutes=int(replace.sum()),
                                                replaced_observed_cells=int(x[replace,:,1].sum()),pred=pred))
                perturb=model.predict(np.stack(inputs))
                for item, value in zip(details,perturb):
                    item.update(perturbed_pred=float(value),signed_change_C=float(value-pred),
                                abs_change_C=float(abs(value-pred)),
                                delta_absolute_error_C=float(abs(value-row.y)-abs(pred-row.y)))
                    records.append(item)
        del model
        torch.cuda.empty_cache()
    frame=pd.DataFrame(records)
    eligible=pd.DataFrame(eligibility)
    frame.to_csv(OUT/'occlusion.csv',index=False)
    eligible.to_csv(OUT/'eligibility.csv',index=False)
    results,person_frames=[],[]
    for scope,subset in [('all_eligible',frame[frame.eligibility.eq('eligible')]),
                         ('at_least3fullblocks',frame[frame.eligibility.eq('eligible') & frame.full_blocks.ge(3)])]:
        for reference,group in subset.groupby('reference'):
            cases=group.groupby(['person','sample_id','fraction','order']).abs_change_C.mean().unstack('order')
            cases['top_minus_random']=cases.top-cases.random
            cases['top_minus_least']=cases.top-cases.least
            by_person=cases.groupby(['person','fraction']).mean().reset_index()
            person_frames.append(by_person.assign(scope=scope,reference=reference))
            for fraction,values in by_person.groupby('fraction'):
                delta=values.top_minus_random.to_numpy()
                rng=np.random.default_rng(RNG_SEED)
                boots=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
                lo,hi=np.quantile(boots,[.025,.975])
                results.append(dict(scope=scope,reference=reference,fraction=fraction,people=len(values),
                                    samples=int(group[group.fraction.eq(fraction)].sample_id.nunique()),
                                    top=float(values.top.mean()),random=float(values.random.mean()),
                                    least=float(values.least.mean()),top_minus_random=float(delta.mean()),
                                    ci_low=float(lo),ci_high=float(hi)))
    pd.concat(person_frames,ignore_index=True).to_csv(OUT/'person_summary.csv',index=False)
    summary=pd.DataFrame(results)
    summary.to_csv(OUT/'summary.csv',index=False)
    completion=dict(status='complete',seconds=time.perf_counter()-started,
                    eligibility_cases=len(eligible),eligible_cases=int(eligible.eligibility.eq('eligible').sum()),
                    not_applicable_cases=int(eligible.eligibility.eq('not_applicable').sum()),
                    source_samples=int(eligible.sample_id.nunique()),source_people=int(eligible.person.nunique()),
                    perturbations=int(frame.eligibility.eq('eligible').sum()),
                    exact_duration_check=bool((frame.loc[frame.eligibility.eq('eligible'),'replaced_minutes']==
                                               5*frame.loc[frame.eligibility.eq('eligible'),'blocks']).all()))
    dump(OUT/'completion.json',completion)
    print(json.dumps(completion))
    print(summary[summary.fraction.eq(.3)].to_string(index=False))


if __name__=='__main__':
    main()
