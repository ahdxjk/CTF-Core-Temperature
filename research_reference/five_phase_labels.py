"""Label PROSPIE activity phases without reading temperature or prediction columns.

Run with the bundled Python runtime. The original workbook is read-only.
The first recorded sample in a new work/rest state is the event time.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"REPLACE_WITH_DATA_DIR\prospie\Prospie dataset_open access V3.xlsx")
OUT = ROOT / "outputs/five_phase_v1/labels"
WINDOWS = (5, 10, 15)
PHASES = {
    "P1": "初始静息期",
    "P2": "运动开始早期",
    "P3": "持续运动期",
    "P4": "运动停止后的早期恢复期",
    "P5": "后续恢复期",
    "U": "阶段无法判定",
}


def minute_value(value):
    if isinstance(value, (datetime, time)):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, timedelta):
        return value.total_seconds() / 60
    if isinstance(value, (float, int)):
        return float(value) * 24 * 60
    raise ValueError(f"Unsupported timestamp {value!r}")


def read_activity():
    # Temperature starts at AD; read only A:X. No core/skin temperature,
    # model predictions, error values, endpoint-temperature change or futures.
    wb = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    ws = wb["Data"]
    header = list(ws.iter_rows(min_row=1, max_row=2, max_col=24, values_only=True))
    names = [str(b or a or "").strip() for a, b in zip(*header)]
    cols = {name: names.index(name) for name in
            ("Condition", "Time", "Activity code", "Participant", "workrest", "Speedkm.hr-1", "Gradient%")}
    records = []
    for excel_row, row in enumerate(ws.iter_rows(min_row=3, max_col=24, values_only=True), 3):
        if row[cols["Participant"]] is None:
            continue
        records.append({
            "source_excel_row": excel_row,
            "Participant": int(row[cols["Participant"]]),
            "Condition": int(row[cols["Condition"]]),
            "minute": minute_value(row[cols["Time"]]),
            "workrest": row[cols["workrest"]],
            "activity_code": row[cols["Activity code"]],
            "treadmill_speed_audit": row[cols["Speedkm.hr-1"]],
            "treadmill_gradient_audit": row[cols["Gradient%"]],
        })
    wb.close()
    d = pd.DataFrame(records)
    d["trial"] = (d.Participant.ne(d.Participant.shift())
                  | d.Condition.ne(d.Condition.shift())
                  | d.minute.le(d.minute.shift())).cumsum()
    return d[["source_excel_row", "Participant", "Condition", "trial",
              "minute", "workrest", "activity_code", "treadmill_speed_audit", "treadmill_gradient_audit"]]


def label_trial(g):
    """Only timestamp/workrest/history determine phase. Activity codes are audit-only.

    A later-starting or first-work recording remains U until a confirmed switch.
    Missing states/time gaps break transition certainty. At the next confirmed
    switch certainty is restored. No missing states are filled.
    """
    out = []
    previous_state = None
    previous_time = None
    initial_rest = False
    transition_time = None
    direction = None
    event_number = 0
    event_id = ""
    observed_work = False
    for i, row in enumerate(g.itertuples(index=False)):
        state = row.workrest
        valid = state in (1, 2)
        continuous = previous_time is not None and np.isclose(row.minute - previous_time, 1)
        if i == 0:
            initial_rest = valid and state == 2 and np.isclose(row.minute, 0)
        elif not continuous:
            previous_state = None
            initial_rest = False
            transition_time = None
            direction = None
            event_id = ""
        changed = valid and previous_state in (1, 2) and state != previous_state
        if changed:
            event_number += 1
            transition_time = float(row.minute)
            direction = "rest_to_work" if state == 1 else "work_to_rest"
            event_id = f"T{int(row.trial):03d}_E{event_number:02d}"
            initial_rest = False
        if not valid:
            initial_rest = False
            transition_time = None
            direction = None
            event_id = ""
        elapsed = (float(row.minute - transition_time)
                   if transition_time is not None else np.nan)
        rec = {
            "is_transition": bool(changed),
            "event_id": event_id,
            "event_direction": direction or "",
            "transition_minute": transition_time,
            "elapsed_since_transition": elapsed,
        }
        if not valid:
            reason = "missing_or_invalid_workrest"
        elif transition_time is None and not initial_rest:
            reason = ("recording_starts_during_work" if i == 0 and state == 1
                      else "state_onset_unobserved")
        else:
            reason = ""
        rec["unknown_reason"] = reason
        for window in WINDOWS:
            if reason:
                phase = "U"
            elif initial_rest and not observed_work:
                phase = "P1"
            elif direction == "rest_to_work":
                phase = "P2" if elapsed < window else "P3"
            elif direction == "work_to_rest":
                phase = "P4" if elapsed < window else "P5"
            else:
                phase = "U"
            rec[f"phase_{window}"] = phase
            rec[f"phase_name_{window}"] = PHASES[phase]
        out.append(rec)
        observed_work = observed_work or (valid and state == 1)
        previous_state = state if valid else None
        previous_time = row.minute
    return pd.DataFrame(out, index=g.index)


def make_segments(d):
    parts = []
    for window in WINDOWS:
        col = f"phase_{window}"
        for trial, g in d.groupby("trial", sort=False):
            chunk = g[col].ne(g[col].shift()).cumsum()
            for n, s in g.groupby(chunk, sort=False):
                parts.append(dict(
                    window_min=window, trial=int(trial),
                    Participant=int(s.Participant.iloc[0]), Condition=int(s.Condition.iloc[0]),
                    segment_id=f"W{window}_T{trial:03d}_S{int(n):02d}",
                    phase=s[col].iloc[0], phase_name=PHASES[s[col].iloc[0]],
                    start_minute=float(s.minute.iloc[0]), end_minute_inclusive=float(s.minute.iloc[-1]),
                    first_source_excel_row=int(s.source_excel_row.iloc[0]),
                    last_source_excel_row=int(s.source_excel_row.iloc[-1]),
                    n_rows=len(s), n_valid_labels=int(s.label_available.sum()),
                    timestamp_span_min=float(s.minute.iloc[-1] - s.minute.iloc[0]),
                    support_duration_min=float(s.support_duration_min.sum()),
                ))
    return pd.DataFrame(parts)


def synthetic_checks():
    # Exact boundaries and repeated switches, independent of recorded results.
    g = pd.DataFrame({"trial": 1, "minute": np.arange(25),
                      "workrest": [2]*3 + [1]*8 + [2]*9 + [1]*5})
    q = label_trial(g)
    assert q.phase_5.iloc[0] == "P1"
    assert q.phase_5.iloc[3] == "P2" and q.phase_5.iloc[7] == "P2"
    assert q.phase_5.iloc[8] == "P3"
    assert q.phase_5.iloc[11] == "P4" and q.phase_5.iloc[15] == "P4"
    assert q.phase_5.iloc[16] == "P5"
    assert q.phase_5.iloc[20] == "P2"
    work_start = pd.DataFrame({"trial": 1, "minute": np.arange(10), "workrest": [1]*7+[2]*3})
    assert label_trial(work_start).phase_5.iloc[:7].eq("U").all()
    assert label_trial(work_start).phase_5.iloc[7:].eq("P4").all()
    gap = pd.DataFrame({"trial": 1, "minute": [0, 1, 4, 5, 6], "workrest": [2, 2, 1, 1, 2]})
    assert label_trial(gap).phase_5.tolist() == ["P1", "P1", "U", "U", "P4"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = read_activity()
    assert len(d) == 17264 and d.Participant.nunique() == 40 and d.trial.nunique() == 155
    assert d.source_excel_row.is_unique
    old = pd.read_csv(ROOT / "outputs/phase_distill_v2/raw_phase_signals.csv")
    common = ["source_excel_row", "Participant", "Condition", "trial", "minute", "workrest"]
    assert len(d) == len(old)
    for col in common:
        assert np.allclose(d[col], old[col]), f"Source alignment mismatch: {col}"
    blocks = [label_trial(g) for _, g in d.groupby("trial", sort=False)]
    d = pd.concat([d, pd.concat(blocks).sort_index()], axis=1)
    # Metadata is used only to mark which rows have an existing usable target;
    # neither its target values nor its validity flags influence the phases.
    meta = pd.read_csv(ROOT / "outputs/cpu_baselines_v1/metadata.csv", usecols=["source_excel_row"])
    d["label_available"] = d.source_excel_row.isin(meta.source_excel_row)
    d["support_duration_min"] = d.groupby("trial").minute.shift(-1).sub(d.minute).fillna(0)
    # Sum supported [t_i,t_{i+1}) intervals, no invented minute after the trial end.
    assert np.isclose(d.support_duration_min.sum(), 17109)
    synthetic_checks()
    checked_prefixes = 0
    for _, g in d.groupby("trial", sort=False):
        for length in sorted(set([1, min(12, len(g)), len(g)//2, len(g)])):
            p = label_trial(g.iloc[:length])
            assert p[[f"phase_{w}" for w in WINDOWS]].equals(g.iloc[:length][[f"phase_{w}" for w in WINDOWS]])
            checked_prefixes += 1
    for w in WINDOWS:
        c = d[f"phase_{w}"]
        assert c.isin(PHASES).all()
        assert d.loc[c.isin(["P2", "P3"]), "workrest"].eq(1).all()
        assert d.loc[c.isin(["P1", "P4", "P5"]), "workrest"].eq(2).all()
        assert d.loc[c.isin(["P2", "P4"]), "elapsed_since_transition"].lt(w).all()
        assert d.loc[c.isin(["P3", "P5"]), "elapsed_since_transition"].ge(w).all()
    activity_expected = {4: 2, 5: 1, 7: 2, 8: 1, 9: 2}
    expected = d.activity_code.map(activity_expected)
    d["activity_workrest_disagreement"] = expected.notna() & d.workrest.ne(expected)
    d["source_review_flag"] = ""
    flag_specs = [
        (13, 1, 83, 83, "isolated_work_marker_without_treadmill_motion"),
        (13, 2, 81, 94, "exercise_period_code_but_rest_and_zero_speed"),
        (10, 3, 81, 81, "outside_rest_code_but_work_and_positive_speed"),
    ]
    for person, condition, start, end, flag in flag_specs:
        mask = d.Participant.eq(person) & d.Condition.eq(condition) & d.minute.between(start, end)
        d.loc[mask, "source_review_flag"] = flag
    review = d[d.source_review_flag.ne("")].copy()
    review["adjudication"] = review.source_review_flag.map({
        "isolated_work_marker_without_treadmill_motion": "Unresolved one-minute inconsistency; preserve workrest in primary labels; exclude trial 115 in sensitivity.",
        "exercise_period_code_but_rest_and_zero_speed": "Treadmill record supports workrest=rest; activity code is coarser/early; primary remains P5.",
        "outside_rest_code_but_work_and_positive_speed": "Treadmill record supports workrest=work; activity code changes one minute later; primary remains P2.",
    })
    review.to_csv(OUT / "review_flags.csv", index=False, encoding="utf-8-sig")
    activity_mismatch = d[d.activity_workrest_disagreement]
    activity_mismatch.to_csv(OUT / "activity_workrest_mismatches.csv", index=False, encoding="utf-8-sig")
    d.to_csv(OUT / "minute_labels.csv", index=False, encoding="utf-8-sig")
    segments = make_segments(d)
    segments.to_csv(OUT / "segments.csv", index=False, encoding="utf-8-sig")
    transitions = d[d.is_transition][["source_excel_row", "Participant", "Condition", "trial", "minute",
                                      "workrest", "activity_code", "event_id", "event_direction"]]
    transitions.to_csv(OUT / "transitions.csv", index=False, encoding="utf-8-sig")
    trials = []
    summary = []
    for w in WINDOWS:
        col = f"phase_{w}"
        for trial, g in d.groupby("trial", sort=False):
            item = dict(window_min=w, trial=int(trial), Participant=int(g.Participant.iloc[0]),
                        Condition=int(g.Condition.iloc[0]), start_minute=float(g.minute.iloc[0]),
                        end_minute=float(g.minute.iloc[-1]), n_rows=len(g), n_valid_labels=int(g.label_available.sum()),
                        rest_to_work_events=int((g.is_transition & g.event_direction.eq("rest_to_work")).sum()),
                        work_to_rest_events=int((g.is_transition & g.event_direction.eq("work_to_rest")).sum()))
            for code in PHASES:
                s = g[g[col].eq(code)]
                item[f"{code}_rows"] = len(s)
                item[f"{code}_valid_labels"] = int(s.label_available.sum())
                item[f"{code}_support_min"] = float(s.support_duration_min.sum())
                item[f"{code}_segments"] = int(((segments.window_min == w) & (segments.trial == trial) & (segments.phase == code)).sum())
            assert sum(item[f"{code}_rows"] for code in PHASES) == len(g)
            trials.append(item)
        for code, name in PHASES.items():
            s = d[d[col].eq(code)]
            seg = segments[(segments.window_min == w) & (segments.phase == code)]
            summary.append(dict(window_min=w, phase=code, phase_name=name,
                                n_people=int(s.Participant.nunique()), n_trials=int(s.trial.nunique()),
                                n_segments=len(seg), n_rows=len(s), n_valid_labels=int(s.label_available.sum()),
                                support_duration_min=float(s.support_duration_min.sum()),
                                row_percent=100*len(s)/len(d)))
    trials = pd.DataFrame(trials)
    summary = pd.DataFrame(summary)
    trials.to_csv(OUT / "trial_phase_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "phase_summary.csv", index=False, encoding="utf-8-sig")
    starts = d.groupby("trial", sort=False).first()
    audit = {
        "source": str(SOURCE), "source_sheet": "Data", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "label_inputs": ["Participant", "Condition", "Time", "workrest"],
        "audit_only_input": "Activity code, treadmill speed/gradient; terminal codes 1/2/3 never trigger events",
        "temperature_or_prediction_columns_read": False,
        "primary_window_min": 5, "sensitivity_windows_min": [10, 15],
        "early_interval": "0 <= minutes_since_first_new_state_sample < window",
        "n_rows": len(d), "n_people": int(d.Participant.nunique()), "n_trials": int(d.trial.nunique()),
        "n_usable_label_rows": int(d.label_available.sum()),
        "all_trials_start_at_minute_zero": bool(starts.minute.eq(0).all()),
        "all_trials_start_in_rest": bool(starts.workrest.eq(2).all()),
        "all_within_trial_gaps_one_minute": bool(d.groupby("trial").minute.diff().dropna().eq(1).all()),
        "workrest_complete": bool(d.workrest.isin([1, 2]).all()),
        "unknown_rows_by_window": {str(w): int(d[f"phase_{w}"].eq("U").sum()) for w in WINDOWS},
        "transition_counts": {str(k): int(v) for k, v in transitions.event_direction.value_counts().items()},
        "activity_workrest_mismatch_rows": len(activity_mismatch),
        "source_review_rows": len(review),
        "strict_unresolved_review_trials": [115],
        "conservative_review_trials": [115, 133, 147],
        "disagreement_interpretation": "Most coarse activity-code disagreements are condition 3 within-period rests. Workrest remains primary; no source values were corrected.",
        "old_source_row_and_trial_alignment_pass": True,
        "prefix_invariance_checks": checked_prefixes, "synthetic_boundary_checks_pass": True,
        "supported_time_span_min": float(d.support_duration_min.sum()),
        "terminal_marker_rows": int(d.activity_code.isin([1, 2, 3]).sum()),
        "terminal_marker_rows_that_also_change_workrest": int((d.activity_code.isin([1, 2, 3]) & d.is_transition).sum()),
        "limitations": [
            "Operational activity phases, not validated physiological states.",
            "5/10/15-minute windows are fixed before the new phase-error comparison, not picked for model improvement.",
            "A phase timestamp is the first one-minute record of the new state; sub-minute switching time is unavailable.",
            "Initial-rest label refers to this experimental record, not proof of no earlier exercise that day.",
            "Recovery here means observed rest following work, not proven physiological recovery.",
            "Phase labels are retrospective study annotations, not deployment inputs.",
            "The same PROSPIE cohort has been repeatedly explored; this is not independent confirmatory validation.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "label_dictionary.json").write_text(json.dumps({
        "phase_names": PHASES,
        "definitions": {
            "P1": "Record starts at t=0 in rest and no work has yet been observed.",
            "P2": "Current state work; observed rest-to-work transition occurred less than w minutes ago.",
            "P3": "Current state work; observed rest-to-work transition occurred at least w minutes ago.",
            "P4": "Current state rest; observed work-to-rest transition occurred less than w minutes ago.",
            "P5": "Current state rest; observed work-to-rest transition occurred at least w minutes ago.",
            "U": "State missing/invalid or onset unknown; no forced phase assignment.",
        },
        "support_duration_min": "Sum of [current timestamp,next timestamp) lengths with current phase; final sample has zero future support.",
        "n_rows": "Number of sampled timestamps, not independent participants or continuous duration.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[summary.window_min.eq(5)].to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
