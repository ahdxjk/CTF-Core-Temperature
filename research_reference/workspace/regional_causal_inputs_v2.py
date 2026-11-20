"""Strict observed-data interface for Regional5 source-only re-training.

The original PROSPIE workbook contains unmarked HR and rectal interpolation
formulas. This module never edits the workbook or legacy raw.pkl. It masks every
formula-derived cell in the five inputs and core target, keeping the original
minute grid and identities. It must be used before normalization or fitting.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/regional_input_causality_audit_v2'
RAW=ROOT/'outputs/raw_tcn_all_v1/raw.pkl'
WORKBOOK=Path('REPLACE_WITH_DATA_DIR/prospie/Prospie dataset_open access V3.xlsx')
COLS=['chest','upper arm','thigh','calf','HR']
TARGET='Corerectal'
ALL22=['abdomen','calf','chest','foot','hand','head','lower arm','lower back','thigh','upper arm','upper back','HR','Ins skin','Tamb','BR','Tfront','Rhfront','Tback','Rhback','Temp','Humidity','Wind']
EXPECTED_SOURCE_ROWS=17264
EXPECTED_LEGACY_LABELS=17162
EXPECTED_STRICT_LABELS=17153


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8'))==value, f'Refusing changed audit overwrite: {path}'
    else:path.write_text(text,encoding='utf8')


def _audit_manifest():
    p=OUT/'audit.json'
    a=json.loads(p.read_text(encoding='utf8'))
    for file,digest in a['artifacts'].items():assert sha(file)==digest,file
    assert a['raw_sha256']==sha(RAW)
    assert a['workbook_sha256']==sha(WORKBOOK)
    return a


def contract():
    a=_audit_manifest()
    return dict(loader_sha256=sha(__file__),audit_sha256=sha(OUT/'audit.json'),
                mask_sha256=sha(OUT/'mask_cells.csv'),raw_sha256=a['raw_sha256'],
                workbook_sha256=a['workbook_sha256'],source_rows=EXPECTED_SOURCE_ROWS,
                legacy_labels=EXPECTED_LEGACY_LABELS,strict_labels=EXPECTED_STRICT_LABELS,
                new_masked_cells={'HR':37,'Corerectal':9},
                rules='Mask all workbook formulas in Regional5+Core irrespective of color; retain all rows; source-only preprocessing; immutable legacy data.')


def clean_frame(frame):
    """Return a copy, masking audited cells by immutable Excel row key.

    Accepts original source, a source subset, or combined source/external frames.
    Negative external row IDs are never touched. When supplied, participant and
    time identities must match the workbook audit. No fitting or imputation.
    """
    _audit_manifest()
    mask=pd.read_csv(OUT/'mask_cells.csv')
    q=frame.copy()
    assert 'source_excel_row' in q
    for r in mask.itertuples():
        where=q.source_excel_row.eq(r.source_excel_row)
        if not where.any():continue
        if 'Participant' in q:assert q.loc[where,'Participant'].eq(r.Participant).all()
        if 'minute' in q:assert np.allclose(q.loc[where,'minute'],r.minute)
        if 'dataset' in q:assert q.loc[where,'dataset'].eq('PROSPIE').all()
        if r.field in q:q.loc[where,r.field]=np.nan
    return q


def load_source_clean():
    """17264 source rows, unchanged RangeIndex and trial IDs, 17153 true labels."""
    original=pd.read_pickle(RAW)
    q=clean_frame(original)
    assert len(q)==EXPECTED_SOURCE_ROWS and q.index.equals(original.index)
    assert q.trial.nunique()==155 and q.Participant.nunique()==40
    assert q.Corerectal.notna().sum()==EXPECTED_STRICT_LABELS
    assert q.HR.notna().sum()==11916
    assert q.groupby('trial').minute.diff().dropna().eq(1).all()
    for c in COLS[:-1]:np.testing.assert_allclose(q[c],original[c],equal_nan=True)
    return q


def load_combined_clean():
    """Clean PROSPIE then unchanged validated Hotwater/Brighton mappings.

    Source trials use the legacy 'prospie_' prefix. External IDs and negative row
    keys remain identical to the legacy transfer experiments. This function
    does not fit normalization or make any target-dependent decision.
    """
    from external_transfer_v1_data import load_hotwater
    from load_brighton_publication_v1 import load_brighton
    d=load_source_clean();d['dataset']='PROSPIE';d['trial']='prospie_'+d.trial.astype(str)
    b=load_brighton()
    for c in ALL22:
        if c not in b:b[c]=np.nan
    b['phase_5']='U'
    q=pd.concat([d,load_hotwater(),b],ignore_index=True)
    assert q[q.dataset.eq('PROSPIE')].Corerectal.notna().sum()==17153
    assert q[q.dataset.eq('Hotwater')].Corerectal.notna().sum()==180
    assert q[q.dataset.eq('Brighton')].Corerectal.notna().sum()==360
    return q


def audit():
    """Build a new, formula-aware provenance audit; never edit legacy files."""
    import openpyxl
    raw=pd.read_pickle(RAW).set_index('source_excel_row')
    wf=openpyxl.load_workbook(WORKBOOK,read_only=True,data_only=False)
    wc=openpyxl.load_workbook(WORKBOOK,read_only=True,data_only=True)
    fi=wf['Data'].iter_rows(max_col=58);ci=wc['Data'].iter_rows(max_col=58)
    a=next(fi);b=next(fi);next(ci);next(ci)
    headers=[str(y.value or x.value or '').strip() for x,y in zip(a,b)]
    columns={c:headers.index(c) for c in COLS+[TARGET]}
    records=[];checked=0;original_mapping_equal=True
    for rn,(fr,cr) in enumerate(zip(fi,ci),3):
        if fr[3].value is None:continue
        assert rn in raw.index
        r=raw.loc[rn];assert r.Participant==cr[3].value
        tm=cr[1].value;minute=tm.hour*60+tm.minute+tm.second/60 if hasattr(tm,'hour') else float(tm)*1440
        assert r.minute==minute;checked+=1
        for name,j in columns.items():
            f,c=fr[j],cr[j]
            color=f.font.color if f.font else None
            red=bool(color and color.type=='rgb' and color.rgb[-6:]=='FF0000')
            expected=np.nan if red or c.value==9999 or c.value is None else float(c.value)
            actual=r[name]
            assert (pd.isna(expected) and pd.isna(actual)) or np.isclose(expected,actual,atol=1e-10,rtol=0),(rn,name)
            if f.data_type!='f':continue
            references=re.findall(r'\$?([A-Z]+)\$?(\d+)',str(f.value))
            rows=[int(v) for _,v in references]
            cross=any(int(raw.loc[k,'trial'])!=int(r.trial) for k in rows if k in raw.index)
            assert not cross
            records.append(dict(field=name,cell=f.coordinate,source_excel_row=rn,formula=str(f.value),
                Participant=int(r.Participant),trial=int(r.trial),minute=float(r.minute),red=red,
                font_type=color.type if color else '',font_value=str(color.value) if color else '',
                cache_value=float(c.value) if c.value is not None else None,
                legacy_raw_value=float(actual) if pd.notna(actual) else None,
                legacy_retained=bool(pd.notna(actual)),referenced_cells=json.dumps([f'{x}{y}' for x,y in references]),
                reference_rows=json.dumps(rows),references_future=max(rows,default=rn)>rn,
                maximum_direct_future_minutes=max(0,max(rows,default=rn)-rn),cross_trial=cross))
    wf.close();wc.close()
    assert checked==17264
    formulas=pd.DataFrame(records)
    mask=formulas[formulas.legacy_retained].copy()
    assert mask.groupby('field').size().to_dict()=={'Corerectal':9,'HR':37}
    assert mask.loc[mask.field.eq('HR'),'references_future'].sum()==33
    assert mask.loc[mask.field.eq('Corerectal'),'source_excel_row'].tolist()==list(range(16823,16832))
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,table in [('all_formula_cells.csv',formulas),('mask_cells.csv',mask),
                       ('hr_formula_cells.csv',mask[mask.field.eq('HR')]),('core_formula_cells.csv',mask[mask.field.eq('Corerectal')])]:
        p=OUT/name
        table.to_csv(p,index=False,encoding='utf8');paths.append(p)
    # External input cells: audit the exact original individual channels used by loaders.
    a=json.loads((ROOT/'outputs/five_phase_v1/external/external_audit.json').read_text(encoding='utf8'))
    hp=next(Path(p) for p in a['sources'] if p.endswith('.xlsx') and 'Hot water immersion' in p)
    hw=openpyxl.load_workbook(hp,read_only=True,data_only=False)
    hs={n:list(hw[n].iter_rows()) for n in ['Arms in vs Arms out','Mean skin & body temp.']}
    points=pd.read_csv(ROOT/'outputs/five_phase_v1/external/hotwater_source_audit_points.csv')
    external=[]
    def checkcell(dataset,field,cell):
        color=cell.font.color if cell.font else None
        external.append(dict(dataset=dataset,field=field,cell=cell.coordinate,
            formula=cell.data_type=='f',red=bool(color and color.type=='rgb' and color.rgb[-6:]=='FF0000'),
            numeric_or_blank=cell.value is None or isinstance(cell.value,(int,float))))
    for p in points.itertuples():
        k=int(p.ID)-1;i=int(p.minute)//5;start,col0=(3,3) if p.condition=='Arms in' else (22,15);row=1+9*k
        checkcell('Hotwater','HR',hs['Arms in vs Arms out'][start+k-1][2+i-1])
        for name,offset in [('chest',2),('upper arm',3),('thigh',4),('calf',5)]:
            checkcell('Hotwater',name,hs['Mean skin & body temp.'][row+offset-1][col0+i-1])
    hw.close()
    bp=ROOT/'outputs/independent_external_audit_v1/Brighton_107.xlsx'
    bw=openpyxl.load_workbook(bp,read_only=True,data_only=False)
    for person in range(1,11):
        rows=list(bw[f'P{person}'].iter_rows())
        for header in [3,24,45]:
            for offset in [2,9]:
                for cells in rows[header+offset:header+offset+6]:
                    for name,j in [('HR',5),('chest',6),('upper arm',7),('thigh',8),('calf',9)]:
                        checkcell('Brighton',name,cells[j])
    bw.close()
    ex=pd.DataFrame(external);assert len(ex)==2700 and not ex.formula.any() and not ex.red.any() and ex.numeric_or_blank.all()
    p=OUT/'external_input_cell_types.csv';ex.to_csv(p,index=False);paths.append(p)
    audit_record=dict(status='passed',scope='Regional5 five inputs and rectal target formula/legacy-cache audit',
        loader_sha256_at_audit=sha(__file__),workbook=str(WORKBOOK),workbook_sha256=sha(WORKBOOK),
        raw_path=str(RAW),raw_sha256=sha(RAW),source_rows=17264,source_mapping_rows_verified=checked,
        cache_values_match_legacy=True,legacy_labels=17162,strict_labels=17153,legacy_HR_observations=11953,strict_HR_observations=11916,
        all_formula_count=len(formulas),new_masked_cells=mask.groupby('field').size().to_dict(),
        HR=dict(cells=37,people=int(mask[mask.field.eq('HR')].Participant.nunique()),
                trials=int(mask[mask.field.eq('HR')].trial.nunique()),future_formula_cells=33,
                maximum_direct_future_minutes=5,past_only_interpolations=3,constant_formula_cells=1),
        core=dict(cells=9,people=[15],trials=[152],minute_range=[45,53],source_excel_rows=list(range(16823,16832))),
        cross_trial_reference_count=int(formulas.cross_trial.sum()),
        external_inputs=dict(Hotwater_cells=900,Brighton_cells=1800,formula_cells=0,red_cells=0),
        outside_scope='No guarantee that numeric cells with undisclosed upstream processing are instantaneous sensor samples. This audit covers the five matched inputs and core, not every possible expanded22 input.',
        rules='All46 previously retained formula cells masked; no reconstruction of underlying readings; all source rows retained; original raw and workbooks unchanged. Legacy17162 and strict17153 scores must not be mixed.',
        external_hashes={str(hp):sha(hp),str(bp):sha(bp)},artifacts={str(p):sha(p) for p in paths})
    _write_json(OUT/'audit.json',audit_record)
    clean=load_source_clean();assert len(clean)==17264
    print(json.dumps(contract(),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','check']);args=p.parse_args()
    import psutil
    psutil.Process().cpu_affinity([0])
    if args.mode=='audit':audit()
    else:
        d=load_combined_clean();print(d.groupby('dataset').Corerectal.count().to_dict())
