"""Read only Brighton P1..P10 local-time rest/exercise blocks. No target fitting or scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs/independent_external_audit_v1"
XLSX=OUT/"Brighton_107.xlsx"
DIGEST="efb4d20b9e8b03397b52c74fe42b75e9a638d5f3d347ca7710703b1e44874a14"
SIGNALS=["Corerectal","HR","chest","upper arm","thigh","calf"]


def load_brighton():
    assert hashlib.sha256(XLSX.read_bytes()).hexdigest()==DIGEST
    wb=load_workbook(XLSX,read_only=True,data_only=True)
    frames=[]; flags=[]
    for i in range(1,11):
        sheet=f"P{i}"
        rows=list(wb[sheet].values)
        for header,arm in zip([3,24,45],["Control","Water","Menthol"]):
            assert str(rows[header-1][1]).strip()==arm
            assert [str(rows[header][j]).strip() for j in [1,2,3,5,6,7,8,9]]==[
                "Time (min)","Power (w)","Core rectal","HR","Chest (°C)","Bicep (°C)","Quads (°C)","Calf (°C)"]
            for phase,offset in [("rest",2),("exercise",9)]:
                begin=header+offset; block=rows[begin:begin+6]
                assert str(block[0][0]).strip().lower()==phase
                assert [r[1] for r in block]==list(range(5,31,5))
                assert all(((r[2]>0) if phase=="exercise" else (r[2]==0)) for r in block)
                points=pd.DataFrame([[r[j] for j in [3,5,6,7,8,9]] for r in block],columns=SIGNALS,
                                    index=pd.Index(range(5,31,5),name="minute"))
                points["source_row"]=np.arange(begin+1,begin+7)
                for sensor in ["chest","upper arm","thigh","calf"]:
                    bad=points[sensor].notna()&~points[sensor].between(10,45)
                    for m,row in points[bad].iterrows():
                        flags.append(dict(sheet=sheet,source_row=int(row.source_row),minute=m,sensor=sensor,value=row[sensor]))
                    points.loc[bad,sensor]=np.nan
                series=points.reindex(pd.Index(range(5,31),name="minute")).reset_index()
                series["source_observed_time"]=series.source_row.notna()
                series["Participant"]=3000+i;series["Condition"]=arm;series["block"]=phase
                series["trial"]=f"{sheet}_{arm}_{phase}";series["session"]=f"{sheet}_{arm}"
                series["dataset"]="Brighton";series["source_sheet"]=sheet
                frames.append(series)
    wb.close()
    d=pd.concat(frames,ignore_index=True)
    d["source_excel_row"]=-300000-d.index.to_numpy()
    assert len(d)==1560 and d.source_observed_time.sum()==360 and d.trial.nunique()==60 and d.Participant.nunique()==10
    assert (d.groupby("trial").minute.diff().dropna()==1).all()
    assert d.loc[~d.source_observed_time,SIGNALS].isna().all().all()
    assert d.loc[d.source_observed_time,["Corerectal","HR","thigh"]].notna().all().all()
    d.attrs["source_flags"]=flags
    return d


if __name__=="__main__":
    d=load_brighton()
    record=dict(workbook_sha256=DIGEST,source_url="https://researchdata.brighton.ac.uk/id/eprint/107/",
                source_doi="10.17033/data.00000107",license="CC BY-ND 3.0 per repository",
                file_scope="10 participant sheets, three named arms each. Ignore portal cohort N=28 for row counts. The named workbook contains cooling-control trial fields.",
                mapping=dict(Corerectal="D: Core rectal (C)",HR="F: HR (bpm)",chest="G: Chest (C)",
                             upper_arm="H: Bicep (C)",thigh="I: Quads (C)",calf="J: Calf (C)"),
                clock="Use observed 5,10,15,20,25,30 min within rest and exercise separately. Prechamber at 0 and untimed post-cognitive testing excluded. No concatenation over rest/exercise boundary. One synthetic clock row per intervening minute holds missing signals. 30-min source-target pairs unavailable.",
                quality="Skin values outside [10,45]C set to missing, no guessed corrections; filter fixed before prediction. Input-only QC.",
                flags=d.attrs["source_flags"],people=10,sessions=30,blocks=60,actual_points=360,minutes=len(d))
    (OUT/"brighton_mapping.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(record)
