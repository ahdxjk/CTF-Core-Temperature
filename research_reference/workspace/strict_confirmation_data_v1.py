"""Immutable expanded formula audit for the primary-claim confirmation run."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
import regional_causal_inputs_v2 as old
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/strict_confirmation_v1'
COLS=old.ALL22
BASE=COLS[:12]
PANELS={'Base12':BASE,'Neck14':BASE+['Ins skin','Tamb']}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def audit():
 if (OUT/'audit.json').exists():return load()
 import openpyxl
 OUT.mkdir(parents=True,exist_ok=True)
 raw=pd.read_pickle(old.RAW);lookup=raw.set_index('source_excel_row')
 wf=openpyxl.load_workbook(old.WORKBOOK,read_only=True,data_only=False)
 wc=openpyxl.load_workbook(old.WORKBOOK,read_only=True,data_only=True)
 fi=wf['Data'].iter_rows();ci=wc['Data'].iter_rows()
 h1,h2=next(fi),next(fi);next(ci);next(ci)
 headers=[str(b.value or a.value or '').strip() for a,b in zip(h1,h2)]
 columns={c:headers.index(c) for c in COLS+['Corerectal']}
 records=[];count=0
 for rn,(fr,cr) in enumerate(zip(fi,ci),3):
  if rn not in lookup.index:continue
  r=lookup.loc[rn];assert r.Participant==cr[3].value;count+=1
  for name,j in columns.items():
   f,c=fr[j],cr[j];color=f.font.color if f.font else None
   red=bool(color and color.type=='rgb' and color.rgb[-6:]=='FF0000')
   val=np.nan if red or c.value is None or c.value==9999 else float(c.value)
   assert (pd.isna(val) and pd.isna(r[name])) or np.isclose(val,r[name],atol=1e-10,rtol=0),(rn,name,val,r[name])
   if f.data_type!='f':continue
   refs=[int(x) for x in re.findall(r'\$?[A-Z]+\$?(\d+)',str(f.value))]
   records.append(dict(field=name,cell=f.coordinate,source_excel_row=rn,Participant=int(r.Participant),trial=int(r.trial),minute=float(r.minute),formula=str(f.value),retained=bool(pd.notna(r[name])),future=bool(max(refs,default=rn)>rn)))
 wf.close();wc.close();assert count==17264
 formulas=pd.DataFrame(records);mask=formulas[formulas.retained].copy()
 formulas.to_csv(OUT/'all_formula_cells.csv',index=False);mask.to_csv(OUT/'mask_cells.csv',index=False)
 d=raw.copy()
 for r in mask.itertuples():d.loc[d.source_excel_row.eq(r.source_excel_row),r.field]=np.nan
 assert d.index.equals(raw.index) and len(d)==17264
 d.to_pickle(OUT/'clean_raw.pkl')
 # Structural upper bound on downstream affected histories, not a measured prediction effect.
 affected=[]
 for panel,cols in PANELS.items():
  m=mask[mask.field.isin(cols)&mask.future]
  starts=m.groupby('trial').minute.min()
  for trial,t in starts.items():
   g=d[d.trial.eq(trial)];a=g[g.minute>=t]
   affected.extend(dict(panel=panel,source_excel_row=int(r.source_excel_row),Participant=int(r.Participant),trial=int(trial),minute=float(r.minute)) for r in a.itertuples())
 pd.DataFrame(affected).to_csv(OUT/'potentially_affected_prefix_origins.csv',index=False)
 files=[old.RAW,old.WORKBOOK,Path(__file__),OUT/'clean_raw.pkl',OUT/'mask_cells.csv']
 save(OUT/'audit.json',dict(status='passed',channels=COLS,rows=len(d),labels=int(d.Corerectal.notna().sum()),people=40,trials=155,new_masked_cells=mask.groupby('field').size().to_dict(),hashes={str(p):sha(p) for p in files},rules='All formula-derived cells in all22+rectal masked before any features, normalization or fit. All timeline rows retained. Numeric-cell upstream preprocessing unknown; no reconstructed labels. Affected-prefix counts are dependency upper bounds, not causal performance changes.'))
 return d
def load():
 a=read(OUT/'audit.json')
 for p,h in a['hashes'].items():assert sha(p)==h,p
 return pd.read_pickle(OUT/'clean_raw.pkl')
if __name__=='__main__':
 audit();print(json.dumps(read(OUT/'audit.json'),ensure_ascii=False,indent=2))
