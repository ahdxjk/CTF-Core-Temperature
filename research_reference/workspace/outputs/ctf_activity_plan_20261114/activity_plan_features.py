"""Causal activity history and one simple, prospectively known activity block.

No model fitting; old experiments are never modified.  Numeric model features
are separated from an auditable, human-readable protocol ledger.  Plan durations
are continuous minutes on [t, t+h), not counts of future sampled state labels.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from revision_schedule_v3 import _first_duration, schedule as reference_schedule

HISTORY_COLUMNS = [
    'activity_current_observed_minutes', 'activity_current_onset_known',
    'activity_previous_work', 'activity_previous_observed_minutes',
    'activity_previous_state_known', 'activity_previous_duration_known',
]
PLAN_COLUMNS = [
    'simple_plan_known', 'simple_plan_work', 'simple_plan_start_delay',
    'simple_plan_minutes_in_horizon', 'simple_plan_end_within_horizon',
]
CONDITIONS = ('History', 'SimplePlan')


def _state(value):
    return int(value) if pd.notna(value) and value in (1, 2) else None


def _blocks(condition, first_onset, final_onset):
    """Documented bounded blocks only; feedback-rest duration is never guessed."""
    if final_onset is not None:
        duration = 25 if condition == 3 else 40
        blocks = [(final_onset, final_onset + duration, 1, 'observed_restart_work')]
        if condition == 3:
            blocks.append((final_onset + 25, final_onset + 35, 0,
                           'documented_final_fixed_rest'))
        return blocks
    anchor = 10.0 if first_onset is None else first_onset
    blocks = [(0.0, anchor, 0, 'initial_fixed_rest')]
    if condition == 3:
        blocks += [(anchor, anchor + 25, 1, 'documented_first_work'),
                   (anchor + 25, anchor + 35, 0, 'documented_mid_fixed_rest'),
                   (anchor + 35, anchor + 60, 1, 'documented_second_work')]
    else:
        blocks.append((anchor, anchor + _first_duration(condition), 1,
                       'documented_first_work'))
    return blocks


def build_context(d: pd.DataFrame) -> pd.DataFrame:
    """Return history features and raw plan metadata for every supplied origin.

    Only trial, its first Condition, minute, and workrest are read. Each trial
    must be a contiguous observed prefix starting at minute zero. No future
    activity, future physiology, outcomes, terminal codes, or record-end time
    are consulted to establish an announced block.
    """
    required = {'trial', 'Condition', 'minute', 'workrest'}
    if not required.issubset(d):
        raise ValueError(f'Missing columns: {sorted(required.difference(d))}')
    if not d.index.is_unique:
        raise ValueError('Unique source index required')
    records = [None] * len(d)
    for positions in d.groupby('trial', sort=False).indices.values():
        g = d.iloc[positions]
        times = g.minute.to_numpy(dtype=float)
        if not len(times) or times[0] != 0 or np.any(np.diff(times) != 1):
            raise ValueError('Each trial must be a contiguous prefix from minute zero')
        condition = int(g.Condition.iloc[0])
        first_duration = _first_duration(condition)
        previous_state = None
        run_start = 0.0
        run_onset_known = False
        previous_run = None
        first_onset = None
        feedback_observed = False
        final_onset = None
        for pos, t, value in zip(positions, times, g.workrest.to_numpy()):
            state = _state(value)
            history = dict.fromkeys(HISTORY_COLUMNS, 0.0)
            if state is None:
                previous_run = None
                run_onset_known = False
                run_start = t
            else:
                if previous_state is None:
                    run_start = t
                    run_onset_known = False
                    previous_run = None
                elif state != previous_state:
                    previous_run = (previous_state, t - run_start, run_onset_known)
                    run_start = t
                    run_onset_known = True
                history['activity_current_observed_minutes'] = t - run_start
                history['activity_current_onset_known'] = float(run_onset_known)
                if previous_run is not None:
                    old_state, duration, duration_known = previous_run
                    history['activity_previous_work'] = float(old_state == 1)
                    history['activity_previous_observed_minutes'] = duration
                    history['activity_previous_state_known'] = 1.0
                    history['activity_previous_duration_known'] = float(duration_known)

            rec = dict(history, origin_minute=float(t), plan_available=False,
                       plan_work=0.0, plan_start=0.0, plan_end=0.0,
                       plan_source='none', plan_reason='unknown_current_activity')
            if state is None:
                records[pos] = rec
                previous_state = None
                continue

            # Same observed-prefix anchoring as the audited R3 protocol ledger.
            if first_onset is None and state == 1:
                if previous_state == 2 or t == 0:
                    first_onset = t
                else:
                    rec['plan_reason'] = 'first_work_onset_unobserved'
                    records[pos] = rec
                    previous_state = state
                    continue
            anchor = 10.0 if first_onset is None else first_onset
            feedback_entry = anchor + (60 if condition == 3 else first_duration)
            if first_onset is not None and t >= feedback_entry and state == 2:
                feedback_observed = True
            if (first_onset is not None and feedback_observed and final_onset is None
                    and t > feedback_entry and previous_state == 2 and state == 1):
                final_onset = t

            if first_onset is None and t >= 10:
                rec['plan_reason'] = 'announced_first_start_passed_without_work'
            elif final_onset is None and t >= feedback_entry:
                rec['plan_reason'] = 'feedback_restart_not_yet_observed'
            else:
                blocks = _blocks(condition, first_onset, final_onset)
                active = next((b for b in blocks if b[0] <= t < b[1]), None)
                if active is None:
                    rec['plan_reason'] = 'outside_documented_bounded_blocks'
                elif active[2] != int(state == 1):
                    rec['plan_reason'] = 'observed_state_conflicts_with_nominal_block'
                else:
                    # Work: its announced remaining duration. Rest: the next
                    # announced work block, if one exists; otherwise fixed rest.
                    chosen = active
                    if active[2] == 0:
                        chosen = next((b for b in blocks if b[2] == 1 and b[0] > t), active)
                    start, end, work, source = chosen
                    rec.update(plan_available=True, plan_work=float(work),
                               plan_start=float(max(t, start)), plan_end=float(end),
                               plan_source=source,
                               plan_reason=('current_declared_block_remaining'
                                            if start <= t else 'next_announced_work_block'))
            records[pos] = rec
            previous_state = state
    result = pd.DataFrame(records, index=d.index)
    if not len(d):
        return pd.DataFrame(columns=HISTORY_COLUMNS + ['origin_minute', 'plan_available',
            'plan_work', 'plan_start', 'plan_end', 'plan_source', 'plan_reason'], index=d.index)
    assert np.isfinite(result[HISTORY_COLUMNS].to_numpy(dtype=float)).all()
    return result


def features_from_context(context: pd.DataFrame, h: int, condition: str) -> pd.DataFrame:
    """Model-only columns; plan timing is clipped to the prediction window.

    An announced work block starting at/after t+h remains marked known, but
    delay is capped at h, overlap is zero and end-within-window is zero. No
    exact timing beyond h is revealed. Unknown plans have all five fields zero.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'Expected one of {CONDITIONS}, got {condition!r}')
    if not isinstance(h, (int, np.integer)) or h <= 0:
        raise ValueError('Positive integer horizon required')
    out = context[HISTORY_COLUMNS].astype(np.float32).copy()
    if condition == 'SimplePlan':
        known = context.plan_available.to_numpy(dtype=bool)
        t = context.origin_minute.to_numpy(dtype=float)
        start = context.plan_start.to_numpy(dtype=float)
        end = context.plan_end.to_numpy(dtype=float)
        overlap = np.maximum(0.0, np.minimum(end, t + h) - np.maximum(start, t))
        out['simple_plan_known'] = known.astype(np.float32)
        out['simple_plan_work'] = np.where(known, context.plan_work, 0).astype(np.float32)
        out['simple_plan_start_delay'] = np.where(known, np.clip(start - t, 0, h), 0).astype(np.float32)
        out['simple_plan_minutes_in_horizon'] = np.where(known, overlap, 0).astype(np.float32)
        out['simple_plan_end_within_horizon'] = (known & (end <= t + h)).astype(np.float32)
    assert np.isfinite(out.to_numpy()).all()
    return out


def build_features(d: pd.DataFrame, h: int, condition: str) -> pd.DataFrame:
    return features_from_context(build_context(d), h, condition)


def protocol_description():
    return {
        'version': 'simple-next-block-20261114-v1',
        'conditions': list(CONDITIONS), 'history_columns': HISTORY_COLUMNS,
        'plan_columns': PLAN_COLUMNS,
        'history': 'Current observed run elapsed; onset observed flag; immediately preceding observed run state/duration and state/duration known flags. First recorded run is left-censored. Missing activity clears both run contexts; no bridging missing states.',
        'plan': 'One announced block: current intended work continuation; from initial/fixed rest the next scheduled work block; at final bounded C3 rest its remaining rest. Current state/protocol conflicts, unknown current activity, feedback recovery before observed restart, and undocumented intervals have no plan.',
        'source': 'Documented R3 nominal protocol blocks anchored only by current/past observed first work onset and feedback restart. Current work/rest uses raw codes1/2. Conditions and trial elapsed locate the nominal block but are never additional model predictors.',
        'time_semantics': 'Physical duration is continuous overlap of the single block with [t,t+h). Example immediate30min work provides overlap30 at h30, while sampled activity at t+30 is outside that half-open block. Start delay cappedh; endwithin flag1 iff declared endpoint<=t+h. Known upcoming work beyondh retains known/type but has delayh/overlap0/endflag0.',
        'unknown': 'Unknown plan five fields all0 includingknown0; no implicit rest filling and no actual future action lookup. Unknown history onset duration is a lower bound with knownflag0.',
        'excluded': ['true current/past/future core temperature', 'future physiological measurements',
                     'realized future workrest', 'terminal/activity event codes',
                     'exact trial end', 'experiment identity as predictor',
                     'nominal first bout as separate predictor', 'complete future schedule',
                     'job type or unverified planned intensity'],
    }


def _synthetic(condition=3, feedback_return=85, n=121):
    t = np.arange(n)
    w = np.full(n, 2.0)
    if condition == 3:
        for a, b in [(10, 35), (45, 70), (feedback_return, feedback_return + 25)]:
            w[(t >= a) & (t < b)] = 1
    else:
        for a, b in [(10, 10 + _first_duration(condition)), (feedback_return, feedback_return + 40)]:
            w[(t >= a) & (t < b)] = 1
    return pd.DataFrame({'trial': 1, 'Condition': condition, 'minute': t, 'workrest': w})


def audit_features(d: pd.DataFrame):
    """All-trial prefix/future-isolation checks and deterministic hand examples."""
    full = build_context(d)
    checks = 0
    for positions in d.groupby('trial', sort=False).indices.values():
        g = d.iloc[positions]
        cuts = sorted({1, min(2, len(g)), min(10, len(g)), min(23, len(g)),
                       max(1, len(g) // 2), max(1, len(g) - 1), len(g)})
        expected = full.iloc[positions]
        for cut in cuts:
            actual = build_context(g.iloc[:cut])
            pd.testing.assert_frame_equal(actual, expected.iloc[:cut])
            checks += 1
        cut = max(1, len(g) // 2)
        changed = g.copy()
        changed.loc[g.index[cut:], 'workrest'] = np.where(
            changed.loc[g.index[cut:], 'workrest'].eq(1), 2, 1)
        changed.loc[g.index[cut:], 'Condition'] = 999
        pd.testing.assert_frame_equal(build_context(changed).iloc[:cut], expected.iloc[:cut])
    # Labels, physiology, IDs and end-reason fields never affect these inputs.
    changed = d.copy()
    for col in changed.columns.difference(['trial', 'Condition', 'minute', 'workrest']):
        changed[col] = 999.0
    pd.testing.assert_frame_equal(build_context(changed), full)

    syn = _synthetic()
    ctx = build_context(syn)
    x = features_from_context(ctx, 30, 'SimplePlan')
    assert x.loc[0, 'simple_plan_start_delay'] == 10
    assert x.loc[0, 'simple_plan_minutes_in_horizon'] == 20
    assert x.loc[10, 'simple_plan_minutes_in_horizon'] == 25
    assert x.loc[34, 'simple_plan_minutes_in_horizon'] == 1
    assert x.loc[35, 'simple_plan_start_delay'] == 10
    assert x.loc[35, 'simple_plan_minutes_in_horizon'] == 20
    assert not ctx.loc[70:84, 'plan_available'].any()
    assert x.loc[85, 'simple_plan_minutes_in_horizon'] == 25
    assert x.loc[110, 'simple_plan_work'] == 0
    assert x.loc[110, 'simple_plan_minutes_in_horizon'] == 10
    assert x.loc[120, 'simple_plan_known'] == 0
    assert ctx.loc[10, 'activity_current_onset_known'] == 1
    assert ctx.loc[10, 'activity_previous_observed_minutes'] == 10
    assert ctx.loc[10, 'activity_previous_duration_known'] == 0
    assert ctx.loc[35, 'activity_previous_observed_minutes'] == 25
    assert ctx.loc[35, 'activity_previous_duration_known'] == 1
    # Extending feedback recovery cannot alter the earlier available plan.
    delayed = _synthetic(feedback_return=100, n=151)
    pd.testing.assert_frame_equal(build_context(delayed).iloc[:85], ctx.iloc[:85])
    missing = syn.copy(); missing.loc[20:21, 'workrest'] = np.nan
    m = build_context(missing)
    assert m.loc[20:21, HISTORY_COLUMNS].eq(0).all().all()
    assert not m.loc[20:21, 'plan_available'].any()
    assert m.loc[22, 'activity_current_onset_known'] == 0
    assert m.loc[22, 'activity_previous_state_known'] == 0
    conflict = syn.copy(); conflict.loc[31:32, 'workrest'] = 2
    c = build_context(conflict)
    assert not c.loc[31:32, 'plan_available'].any()
    assert c.loc[33, 'plan_end'] == 35
    # Declared work immediately starting for30min uses physical duration30.
    worker = ctx.iloc[[85]].copy()
    worker.loc[:, 'plan_end'] = worker.origin_minute + 30
    assert features_from_context(worker, 30, 'SimplePlan').simple_plan_minutes_in_horizon.iloc[0] == 30
    # Known future work outsideh keeps only capped timing, no later schedule.
    early = features_from_context(ctx.iloc[[0]], 5, 'SimplePlan').iloc[0]
    assert early.simple_plan_known == 1 and early.simple_plan_start_delay == 5
    assert early.simple_plan_minutes_in_horizon == 0 and early.simple_plan_end_within_horizon == 0
    # Every disclosed block agrees with existing R3 intention at sampled times
    # inside that selected block. Outside it no additional path is supplied.
    nominal, _ = reference_schedule(d)
    overlap_checks = 0
    for step in range(1, 31):
        time = full.origin_minute.to_numpy() + step
        mask = (full.plan_available.to_numpy(dtype=bool)
                & (time >= full.plan_start.to_numpy()) & (time < full.plan_end.to_numpy()))
        np.testing.assert_array_equal(nominal[mask, step - 1], full.plan_work.to_numpy()[mask])
        overlap_checks += int(mask.sum())
    dimensions = {}
    for h in (5, 10, 15, 30):
        hist = features_from_context(full, h, 'History')
        planned = features_from_context(full, h, 'SimplePlan')
        pd.testing.assert_frame_equal(hist, planned[HISTORY_COLUMNS])
        assert planned.simple_plan_minutes_in_horizon.between(0, h).all()
        assert planned.simple_plan_start_delay.between(0, h).all()
        dimensions[str(h)] = {'History': len(hist.columns), 'SimplePlan': len(planned.columns)}
    return {'status': 'passed', 'rows': len(d), 'trials': int(d.trial.nunique()),
            'prefix_checks': checks, 'all_trial_future_activity_perturbations': True,
            'target_and_other_field_isolation': True, 'hand_examples': 'passed',
            'same_history_both_conditions': True, 'dimensions': dimensions,
            'reference_nominal_block_checks': overlap_checks,
            'plan_available_origins': int(full.plan_available.sum()),
            'reasons': {str(k): int(v) for k, v in full.plan_reason.value_counts().items()}}


if __name__ == '__main__':
    import json
    from strict_confirmation_data_v1 import load
    print(json.dumps(audit_features(load()), indent=2, ensure_ascii=False))
