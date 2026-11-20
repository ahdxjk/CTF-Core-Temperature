"""Public hot-water transfer data; preserve only actual five-minute observations.

This is a region-matched passive-heating stress test, not an exact device/site
replication and not a five-activity-phase validation dataset. No interpolation,
forward filling, mean-skin formula or target-derived predictor is done here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'outputs/five_phase_v1/external'
POINTS = AUDIT_DIR / 'hotwater_source_audit_points.csv'
ALL22 = ['abdomen', 'calf', 'chest', 'foot', 'hand', 'head', 'lower arm',
         'lower back', 'thigh', 'upper arm', 'upper back', 'HR', 'Ins skin',
         'Tamb', 'BR', 'Tfront', 'Rhfront', 'Tback', 'Rhback', 'Temp',
         'Humidity', 'Wind']
REGION_MAP = {'HR': 'HR', 'chest': 'chest', 'bicep': 'upper arm',
              'thigh': 'thigh', 'gastroc': 'calf'}
PANELS = {'ThighHR2': ['thigh', 'HR'],
          'Regional5': ['chest', 'upper arm', 'thigh', 'calf', 'HR']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_workbook():
    audit = json.loads((AUDIT_DIR / 'external_audit.json').read_text(encoding='utf-8'))
    paths = [(Path(p), digest) for p, digest in audit['sources'].items()
             if Path(p).suffix == '.xlsx' and 'Hot water immersion' in p]
    assert len(paths) == 1
    path, expected = paths[0]
    assert path.is_file() and sha(path) == expected, 'Original hot-water source changed'
    return path


def observed_points():
    """Verify every relevant value against the original workbook before returning."""
    original = source_workbook()
    q = pd.read_csv(POINTS, dtype={'ID': str})
    assert len(q) == 180 and q.ID.nunique() == 9
    assert not q.duplicated(['ID', 'condition', 'minute']).any()
    assert set(q.condition) == {'Arms in', 'Arms out'}
    assert q.groupby(['ID', 'condition']).size().eq(10).all()
    book = openpyxl.load_workbook(original, read_only=True, data_only=True)
    # Read sheet values once; read-only random cell lookups would repeatedly parse XML.
    sheets = {name: list(book[name].values)
              for name in ['Arms in vs Arms out', 'Mean skin & body temp.']}
    book.close()

    def cell(sheet, row, col):
        return sheets[sheet][row - 1][col - 1]

    for r in q.itertuples():
        pid = int(r.ID); k = pid - 1; i = int(r.minute) // 5
        assert r.minute == 5*i and 0 <= i < 10
        first_row, col0 = (3, 3) if r.condition == 'Arms in' else (22, 15)
        hrow = 1 + 9*k
        assert str(cell('Arms in vs Arms out', first_row+k, 1)).strip().zfill(3) == r.ID
        assert cell('Mean skin & body temp.', hrow, col0+i) == r.minute
        values = {'HR': cell('Arms in vs Arms out', first_row+k, 2+i),
                  'Tr': cell('Arms in vs Arms out', first_row+k, 35+i)}
        assert values['Tr'] == cell('Mean skin & body temp.', hrow+1, col0+i)
        for name, offset in [('chest', 2), ('bicep', 3), ('thigh', 4), ('gastroc', 5)]:
            values[name] = cell('Mean skin & body temp.', hrow+offset, col0+i)
        for name, value in values.items():
            saved = getattr(r, name)
            if value is None:
                assert pd.isna(saved), (r.ID, r.condition, r.minute, name)
            else:
                assert np.isclose(float(saved), float(value), atol=1e-10, rtol=0), name
    return q


def load_hotwater():
    """Return 828 chronological state rows, with 180 real rectal labels.

    Numeric participant IDs 1001..1009 are disjoint from PROSPIE. Negative source
    row IDs are stable external keys, not Excel row numbers. All other signal
    columns remain missing. Generated minutes are unobserved state-update times,
    never new labels or new measurements. Models may encode history causally.
    """
    observed = observed_points()
    frames = []
    for (pid, condition), g in observed.groupby(['ID', 'condition'], sort=True):
        g = g.set_index('minute').sort_index()
        assert g.index.tolist() == list(range(0, 46, 5))
        f = pd.DataFrame(index=pd.RangeIndex(46, name='minute'))
        for name in ALL22:
            f[name] = np.nan
        for source, target in REGION_MAP.items():
            f[target] = g[source].reindex(f.index)
        f['Corerectal'] = g.Tr.reindex(f.index)
        f['Participant'] = 1000 + int(pid)
        f['external_id'] = pid
        f['Condition'] = condition
        f['trial'] = f"hotwater_{pid}_{condition.lower().replace(' ', '_')}"
        f['phase_5'] = 'U'
        f['dataset'] = 'Hotwater'
        f['source_observed_time'] = f.index.isin(g.index)
        f['source_hr_cell'] = g.source_hr_cell.reindex(f.index)
        f['source_tr_cell'] = g.source_tr_cell.reindex(f.index)
        frames.append(f.reset_index())
    d = pd.concat(frames, ignore_index=True)
    d['source_excel_row'] = -100000 - d.index.to_numpy()
    assert len(d) == 828 and d.trial.nunique() == 18 and d.Participant.nunique() == 9
    assert d.Corerectal.notna().sum() == 180 and d.HR.notna().sum() == 180
    assert d.groupby('trial').minute.diff().dropna().eq(1).all()
    assert d.loc[~d.source_observed_time, ALL22 + ['Corerectal']].isna().all().all()
    assert d[['chest', 'upper arm', 'thigh', 'calf']].notna().sum().tolist() == [155, 176, 180, 180]
    d.attrs['source_manifest'] = manifest()
    return d


def manifest():
    original = source_workbook()
    return {
        'dataset': 'Hotwater', 'source_url': 'https://data.mendeley.com/datasets/tpvm2t62wr/1',
        'source_xlsx': str(original), 'source_sha256': sha(original),
        'audited_points_csv': str(POINTS), 'points_sha256': sha(POINTS),
        'audit_sha256': sha(AUDIT_DIR / 'external_audit.json'),
        'loader_sha256': sha(__file__), 'people': 9, 'trials': 18,
        'observed_points': 180, 'state_rows': 828, 'source_interval_minutes': 5,
        'model_clock_minutes': 1, 'duration_minutes': 45, 'panels': PANELS,
        'anatomical_mapping': REGION_MAP,
        'mapping_limit': 'Bicep/upper arm and gastroc/calf are regional matches, not verified exact site/side/device equivalence.',
        'label': 'Tr (rectal temperature, Celsius); never interpolated or provided as input/initialization.',
        'input_limit': 'No forehead, neck insulation pair, or respiration. MST/MBT and target-derived columns excluded.',
        'phase_limit': 'Passive immersion; no aligned exercise/recovery events, do not force P1-P5 labels.',
        'validation_limit': 'External passive-heating distribution-shift stress test, not proof of P5 recovery generalization.',
    }


if __name__ == '__main__':
    d = load_hotwater()
    print(json.dumps(manifest(), ensure_ascii=False, indent=2))
    print(d.groupby('Condition')[['Corerectal', 'HR', 'chest', 'upper arm', 'thigh', 'calf']].count())
