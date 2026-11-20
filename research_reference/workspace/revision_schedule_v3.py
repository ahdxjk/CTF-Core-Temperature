"""Origin-causal protocol ledger for the third manuscript revision.

Independent of every frozen training implementation. This file does not fit
models and never edits historical predictions. Public API: schedule(frame)
returns (n x 30 nominal activity, four-column clock DataFrame), with -1 unknown,
0 rest and 1 work. Only trial, Condition, minute and workrest are read.

Intervals are half-open: [onset, onset + prescribed duration). Fixed rest in
Condition 3 is known from the protocol. The feedback-rest return is not known
until a rest-to-work transition has actually been observed in the prefix.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/revision_round3_20261108'
RAW = ROOT / 'outputs/strict_confirmation_v1/clean_raw.pkl'
HORIZON = 30
FAMILIES = {'continuous60': [1, 2, 11], 'intermittent': [3],
            'continuous40': [6, 7, 8, 9, 10]}
PAPER_ID = {1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
CLOCK_COLUMNS = ['nominal_first_bout', 'elapsed', 'state_age', 'past_work_bouts']
SOURCE_URL = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8144153/'


def _first_duration(condition: int) -> int:
    if condition == 3:
        return 25
    if condition in FAMILIES['continuous60']:
        return 60
    if condition in FAMILIES['continuous40']:
        return 40
    raise ValueError(f'Unmapped source Condition {condition}')


def _pre_feedback(times: np.ndarray, anchor: float, condition: int) -> np.ndarray:
    """Intended schedule from first work onset to feedback-rest ENTRY only."""
    p = np.full(times.shape, -1, dtype=np.float32)
    duration = _first_duration(condition)
    p[times < anchor] = 0
    if condition == 3:
        p[(times >= anchor) & (times < anchor + 25)] = 1
        p[(times >= anchor + 25) & (times < anchor + 35)] = 0
        p[(times >= anchor + 35) & (times < anchor + 60)] = 1
        p[times == anchor + 60] = 0
    else:
        p[(times >= anchor) & (times < anchor + duration)] = 1
        p[times == anchor + duration] = 0
    return p


def _post_feedback(times: np.ndarray, anchor: float, condition: int) -> np.ndarray:
    p = np.full(times.shape, -1, dtype=np.float32)
    duration = 25 if condition == 3 else 40
    p[(times >= anchor) & (times < anchor + duration)] = 1
    if condition == 3:
        p[(times >= anchor + 25) & (times < anchor + 35)] = 0
    return p


def schedule(d: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Construct intended activities using each origin's observed prefix only.

    Protocol is taken from the trial's first supplied row. Each trial is expected
    to begin at minute zero; a truncated suffix is rejected because its past
    cannot be reconstructed. A prefix ending before a future target is fully
    supported: future row existence never censors the plan.

    The first observed work onset anchors the known pre-feedback protocol. Until
    it is observed, its planned time is minute 10. Condition-3 within-cycle short
    interruptions do not create a new protocol work bout. Fixed blocks retain
    the first-onset clock, rather than fitting their durations to actual future
    switches. Feedback recovery must have been observed at/after its prescribed
    entry and a subsequent adjacent rest-to-work transition observed to anchor
    the last work bout. Unknown current activity gives an unknown future plan.
    """
    required = {'trial', 'Condition', 'minute', 'workrest'}
    if not required.issubset(d):
        raise ValueError(f'Missing columns: {required.difference(d)}')
    nominal = np.full((len(d), HORIZON), -1, dtype=np.float32)
    clock = np.zeros((len(d), 4), dtype=np.float32)
    for positions in d.groupby('trial', sort=False).indices.values():
        g = d.iloc[positions]
        condition = int(g.Condition.iloc[0])
        duration = _first_duration(condition)
        times = g.minute.to_numpy(dtype=float)
        if times[0] != 0 or np.any(np.diff(times) != 1):
            raise ValueError('Each supplied trial must be a contiguous prefix from minute zero')
        previous = None
        state_start = 0.
        bouts = 0
        first_onset = None
        feedback_observed = False
        final_onset = None
        for pos, t, value in zip(positions, times, g.workrest.to_numpy()):
            state = int(value) if pd.notna(value) and value in (1, 2) else None
            if state != previous:
                state_start = t
                bouts += int(state == 1)
            clock[pos] = [duration, t, t - state_start, bouts]
            if state is None:
                previous = None
                continue
            if first_onset is None and state == 1:
                # No reconstruction of an onset across missing activity records.
                if previous == 2 or t == 0:
                    first_onset = t
                else:
                    previous = state
                    continue
            anchor = 10. if first_onset is None else first_onset
            feedback_entry = anchor + (60 if condition == 3 else duration)
            if first_onset is not None and t >= feedback_entry and state == 2:
                feedback_observed = True
            if (first_onset is not None and feedback_observed and final_onset is None
                    and t > feedback_entry and previous == 2 and state == 1):
                final_onset = t
            future = t + np.arange(1, HORIZON + 1)
            if final_onset is not None:
                nominal[pos] = _post_feedback(future, final_onset, condition)
            elif first_onset is None and t >= 10:
                # The announced first onset has passed without observed work.
                pass
            elif t < feedback_entry:
                nominal[pos] = _pre_feedback(future, anchor, condition)
            # In feedback recovery, and after a missed planned boundary, unknown.
            previous = state
    return nominal, pd.DataFrame(clock, columns=CLOCK_COLUMNS, index=d.index)


def _legacy_schedule(d):
    tree = ast.parse((ROOT / 'supplement_forecast_controls.py').read_text(encoding='utf-8'))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'schedule')
    ns = {'np': np, 'pd': pd, 'FAMILIES': FAMILIES}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy_schedule', 'exec'), ns)
    return ns['schedule'](d)


def _synthetic(condition, n=121, feedback_return=85):
    t = np.arange(n)
    a = np.full(n, 2.)
    first = _first_duration(condition)
    if condition == 3:
        a[(t >= 10) & (t < 35)] = 1
        a[(t >= 45) & (t < 70)] = 1
        a[(t >= feedback_return) & (t < feedback_return + 25)] = 1
    else:
        a[(t >= 10) & (t < 10 + first)] = 1
        a[(t >= feedback_return) & (t < feedback_return + 40)] = 1
    return pd.DataFrame({'trial': 1, 'Condition': condition, 'minute': t, 'workrest': a})


def check_hand_examples():
    checked = []
    for condition in PAPER_ID:
        d = _synthetic(condition)
        p, c = schedule(d)
        assert p[0, 8] == 0 and p[0, 9] == 1  # minutes 9 / 10
        assert c.nominal_first_bout.iloc[0] == _first_duration(condition)
        entry = 70 if condition == 3 else 10 + _first_duration(condition)
        assert p[entry - 1, 0] == 0 and p[entry - 1, 1] == -1
        assert np.all(p[entry] == -1)
        assert p[84, 0] == -1 and p[85, 0] == 1  # future return unknown at 84
        checked.append(f'condition_{condition}_onset_feedback_and_observed_return')
    d = _synthetic(3)
    p, _ = schedule(d)
    assert list(p[30, :15]) == [1] * 4 + [0] * 10 + [1]
    assert p[34, 0] == 0 and p[44, 0] == 1
    assert np.all(p[109, :10] == 0) and p[109, 10] == -1
    checked.append('C3_half_open_25_10_25_and_final_25_10')
    delayed = _synthetic(3, n=151, feedback_return=100)
    a, _ = schedule(delayed)
    assert np.array_equal(p[:85], a[:85])
    assert np.all(a[99] == -1) and a[100, 0] == 1
    checked.append('feedback_return_not_anticipated_or_imputed')
    paused = d.copy(); paused.loc[31:32, 'workrest'] = 2
    q, _ = schedule(paused)
    np.testing.assert_array_equal(q, p)
    checked.append('short_within_cycle_interruption_does_not_renumber_protocol_blocks')
    missing = d.copy(); missing.loc[20, 'workrest'] = np.nan
    q, _ = schedule(missing)
    assert np.all(q[20] == -1)
    np.testing.assert_array_equal(q[21:], p[21:])
    checked.append('unknown_current_activity_is_explicitly_unknown')
    # A terminated prefix keeps its previously announced intentions beyond EOF.
    q, _ = schedule(d.iloc[:23])
    np.testing.assert_array_equal(q, p[:23])
    assert np.any(q[-1] >= 0)
    checked.append('no_actual_end_time_censoring')
    return checked


def coverage(d, plans, version):
    rows = []
    for horizon in (5, 15, 30):
        valid = np.zeros(len(d), dtype=bool)
        for ix in d.groupby('trial', sort=False).indices.values():
            if len(ix) > horizon:
                valid[ix[:-horizon]] = d.Corerectal.iloc[ix[horizon:]].notna().to_numpy()
        for condition in [0, *PAPER_ID]:
            ix = np.flatnonzero(valid & (True if condition == 0 else d.Condition.eq(condition)))
            p = plans[ix, :horizon]
            known = p >= 0
            current = (d.workrest.iloc[ix].to_numpy() == 1).astype(np.int8)
            path = np.column_stack((current, p))
            mask = np.column_stack((np.ones(len(ix), bool), known))
            adjacent = mask[:, :-1] & mask[:, 1:]
            wr = adjacent & (path[:, :-1] == 1) & (path[:, 1:] == 0)
            rw = adjacent & (path[:, :-1] == 0) & (path[:, 1:] == 1)
            rows.append(dict(version=version, horizon=horizon, Condition=condition,
                             origins=len(ix), people=int(d.Participant.iloc[ix].nunique()),
                             full=int(known.all(1).sum()), partial=int((known.any(1) & ~known.all(1)).sum()),
                             none=int((~known.any(1)).sum()), known_steps=int(known.sum()),
                             known_work_to_rest_origins=int(wr.any(1).sum()),
                             known_rest_to_work_origins=int(rw.any(1).sum())))
    return pd.DataFrame(rows)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_pickle(RAW)
    hand = check_hand_examples()
    p, c = schedule(d)
    checks = []
    for trial, ix in d.groupby('trial', sort=False).indices.items():
        g = d.iloc[ix].reset_index(drop=True)
        a, b = schedule(g)
        # All prefixes, including all fixed and feedback transitions.
        for cut in range(len(g)):
            pp, cc = schedule(g.iloc[:cut + 1])
            np.testing.assert_array_equal(pp, a[:cut + 1])
            np.testing.assert_array_equal(cc.to_numpy(), b.iloc[:cut + 1].to_numpy())
        for cut in sorted(set([0, 9, 20, min(69, len(g)-1), len(g)//2, len(g)-1])):
            if cut >= len(g):
                continue
            changed = g.copy()
            changed.loc[cut+1:, 'workrest'] = 3 - changed.loc[cut+1:, 'workrest']
            changed.loc[cut+1:, 'Condition'] = 3
            changed.loc[cut+1:, 'Corerectal'] = 900
            changed.loc[cut+1:, 'Activity code'] = 1
            aa, bb = schedule(changed)
            np.testing.assert_array_equal(aa[:cut+1], a[:cut+1])
            np.testing.assert_array_equal(bb.iloc[:cut+1], b.iloc[:cut+1])
        checks.append(dict(trial=int(trial),prefixes_checked=len(g),future_perturbation=True))
    # Target, feedback-event and termination codes never enter nominal inputs.
    minimal = d[['trial', 'Condition', 'minute', 'workrest']].copy()
    pp, cc = schedule(minimal)
    np.testing.assert_array_equal(pp, p); np.testing.assert_array_equal(cc, c)
    old, _ = _legacy_schedule(d)
    cv = pd.concat([coverage(d, old, 'legacy_v2'), coverage(d, p, 'protocol_v3')])
    cv.to_csv(OUT / 'schedule_coverage_old_vs_v3.csv', index=False)
    new = cv[cv.version.eq('protocol_v3')].set_index(['horizon','Condition'])
    prior = cv[cv.version.eq('legacy_v2')].set_index(['horizon','Condition'])
    delta = new.drop(columns='version') - prior.drop(columns='version')
    delta.to_csv(OUT / 'schedule_coverage_delta.csv')
    ledger = pd.concat([d[['source_excel_row','Participant','Condition','trial','minute','workrest']], c], axis=1)
    for j in range(HORIZON):
        ledger[f'nominal_{j+1:02d}'] = p[:, j].astype(np.int8)
        ledger[f'known_{j+1:02d}'] = (p[:, j] >= 0).astype(np.int8)
    ledger.to_csv(OUT / 'schedule_v3_ledger.csv.gz', index=False)
    # Actual future actions are used below only for a separate adherence audit.
    disagreements = []
    for trial, ix in d.groupby('trial', sort=False).indices.items():
        for j in range(HORIZON):
            h = j + 1
            if len(ix) <= h:
                continue
            origin, target = ix[:-h], ix[h:]
            actual = (d.workrest.iloc[target].to_numpy() == 1).astype(np.int8)
            for z in np.flatnonzero((p[origin,j] >= 0) & (p[origin,j] != actual)):
                o, tar = origin[z], target[z]
                disagreements.append(dict(source_excel_row=int(d.source_excel_row.iloc[o]),trial=int(trial),
                    Condition=int(d.Condition.iloc[o]),Participant=int(d.Participant.iloc[o]),
                    origin_minute=float(d.minute.iloc[o]),horizon=h,target_minute=float(d.minute.iloc[tar]),
                    nominal=int(p[o,j]),actual=int(actual[z])))
    pd.DataFrame(disagreements).to_csv(OUT/'schedule_nominal_actual_disagreement_DIAGNOSTIC_ONLY.csv',index=False)
    info = dict(status='passed',module='revision_schedule_v3.py',module_sha256=_sha(__file__),
                source_sha256=_sha(RAW),rows=len(d),labels=int(d.Corerectal.notna().sum()),
                input_columns=list(minimal),hand_examples=hand,all_prefixes_checked=sum(x['prefixes_checked'] for x in checks),
                trials_checked=len(checks),future_perturbation=True,target_and_event_code_exclusion=True,
                nominal_not_guaranteed_adherence=True,training_started=False,
                boundary='half-open [onset,onset+duration); no fitted one-minute offset; feedback entry itself rest, later times unknown',
                C3_resolution='Workbook describes5+5+5; PMC8144153 Table2 and experimental procedures explicitly5+5+15=25. Fixed10-minute in-chamber rest restored.',
                primary_source=SOURCE_URL,conditions=sorted(PAPER_ID),paper_condition_mapping=PAPER_ID)
    (OUT/'schedule_v3_preflight.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'schedule_v3_prefix_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
    print(cv[cv.Condition.isin([0,3])].to_string(index=False))


if __name__ == '__main__':
    audit()
