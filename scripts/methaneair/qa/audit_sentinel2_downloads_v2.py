#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

SLOTS = ['t0','t90','t360']
EXPECTED_BANDS = ['B02','B03','B04','B08','B11','B12']
CLEAR_SCL = {4,5,6,7}

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,default=Path('/project/6002520/yunjung1/MethaneFuse'))
    p.add_argument('--qa-threshold',type=float,default=0.8)
    p.add_argument('--strict-t0-hours',type=float,default=72.0)
    p.add_argument('--patch-pixels',type=int,default=128)
    p.add_argument('--manifest',type=Path,default=None,help='Specific manifest CSV to audit')
    p.add_argument('--output-prefix',default='sentinel2_v2',help='Prefix for audit output CSV files')
    return p.parse_args()

def read_retry(path):
    last=None
    for i in range(5):
        try:return pd.read_csv(path)
        except Exception as e:last=e; time.sleep(i+1)
    raise RuntimeError(f'Cannot read {path}: {last}')

def choose_manifest(root,supplied=None):
    if supplied is not None:
        p=supplied.expanduser()
        if not p.is_absolute(): p=root/p
        if not p.exists(): raise FileNotFoundError(f'Manifest not found: {p}')
        return p,read_retry(p)
    ps=[root/'data/methaneair_full/sentinel2_temporal_manifest_best_qa.partial.csv',
        root/'data/methaneair_full/sentinel2_temporal_manifest_best_qa.csv']
    got=[]
    for p in ps:
        if p.exists():
            try: got.append((p,read_retry(p)))
            except Exception: pass
    if not got: raise FileNotFoundError('No best-QA manifest found')
    return max(got,key=lambda x:len(x[1]))

def resolve(v,root):
    if pd.isna(v) or not str(v).strip(): return None
    p=Path(str(v))
    return p if p.is_absolute() else root/p

def hav(lon1,lat1,lon2,lat2):
    r=6371008.8
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

def center_offset(ds,lon,lat):
    x,y=ds.transform*(ds.width/2,ds.height/2)
    lo,la=warp_transform(ds.crs,'EPSG:4326',[x],[y])
    return hav(lon,lat,float(lo[0]),float(la[0]))

def ttuple(t): return tuple(round(float(x),9) for x in t)

def audit_slot(row,slot,root,a):
    out={'record_id':row.get('record_id'),'label':row.get('label'),'slot':slot,
         'manifest_status':row.get(f'{slot}_status'),'technical_pass':False,
         'qa_pass_recomputed':False,'strict_t0_aligned':np.nan,'error':''}
    ip=resolve(row.get(f'{slot}_path'),root); sp=resolve(row.get(f'{slot}_scl_path'),root)
    out['image_path']=str(ip) if ip else ''; out['scl_path']=str(sp) if sp else ''
    try:
        if ip is None or not ip.exists(): raise FileNotFoundError(f'image missing: {ip}')
        if sp is None or not sp.exists(): raise FileNotFoundError(f'SCL missing: {sp}')
        lon=float(row['longitude']); lat=float(row['latitude'])
        with rasterio.open(ip) as ds:
            out['band_count']=ds.count; out['height']=ds.height; out['width']=ds.width
            out['crs']=str(ds.crs); out['band_descriptions']='|'.join([x or '' for x in ds.descriptions])
            arr=ds.read(out_dtype='float32')
            finite=np.isfinite(arr); out['finite_fraction']=float(finite.mean())
            out['nonconstant_band_count']=int(sum(np.nanstd(b[np.isfinite(b)])>1e-7 for b in arr if np.isfinite(b).any()))
            vals=arr[finite]
            if vals.size:
                out['p01']=float(np.percentile(vals,1)); out['median']=float(np.median(vals)); out['p99']=float(np.percentile(vals,99))
            out['center_offset_m']=center_offset(ds,lon,lat)
            ishape=(ds.height,ds.width); icrs=ds.crs; itr=ttuple(ds.transform)
        with rasterio.open(sp) as ds:
            scl=ds.read(1)
            out['grid_matches_scl']=bool(ds.count==1 and (ds.height,ds.width)==ishape and ds.crs==icrs and ttuple(ds.transform)==itr)
        valid=scl>0; clear=np.isin(scl,list(CLEAR_SCL))
        out['scl_valid_fraction']=float(valid.mean())
        cf=float(clear[valid].mean()) if valid.any() else 0.0
        out['clear_fraction_recomputed']=cf; out['qa_pass_recomputed']=bool(cf>=a.qa_threshold and out['scl_valid_fraction']>=a.qa_threshold)
        md=pd.to_numeric(pd.Series([row.get(f'{slot}_time_delta_hours')]),errors='coerce').iloc[0]
        out['time_delta_hours']=md; out['time_metadata_present']=bool(pd.notna(md))
        if slot=='t0': out['strict_t0_aligned']=bool(pd.notna(md) and abs(float(md))<=a.strict_t0_hours)
        out['technical_pass']=bool(out['band_count']==6 and out['height']==a.patch_pixels and out['width']==a.patch_pixels and out['crs'] not in ('','None') and out['band_descriptions']=='|'.join(EXPECTED_BANDS) and out['finite_fraction']>=0.5 and out['nonconstant_band_count']>=5 and out['grid_matches_scl'] and out['scl_valid_fraction']>=0.8 and out['center_offset_m']<=40)
    except Exception as e: out['error']=f'{type(e).__name__}: {e}'
    return out

def main():
    a=args(); root=a.project_root.expanduser().resolve(); data=root/'data/methaneair_full'
    mp,m=choose_manifest(root,a.manifest)
    rows=[]
    for _,r in m.iterrows():
        for s in SLOTS: rows.append(audit_slot(r,s,root,a))
    audit=pd.DataFrame(rows)
    rec=[]
    for rid,g in audit.groupby('record_id',sort=False):
        d={s:g[g.slot==s].iloc[0] for s in SLOTS if not g[g.slot==s].empty}
        all3=len(d)==3
        tech=all3 and all(bool(d[s].technical_pass) for s in SLOTS)
        qa=all3 and all(bool(d[s].qa_pass_recomputed) for s in SLOTS)
        aligned=all3 and bool(d['t0'].strict_t0_aligned)
        rec.append({'record_id':rid,'label':g.label.iloc[0],'all_three_technical_pass':tech,'all_three_qa_pass_recomputed':qa,'strict_t0_aligned_72h':aligned,'strict_model_ready':bool(tech and qa and aligned)})
    rec=pd.DataFrame(rec)
    audit_path=data/f'{a.output_prefix}_integrity_audit.csv'
    rec_path=data/f'{a.output_prefix}_record_readiness.csv'
    summary_path=data/f'{a.output_prefix}_integrity_summary.csv'
    audit.to_csv(audit_path,index=False)
    rec.to_csv(rec_path,index=False)
    summary=pd.DataFrame([{
        'manifest_used':str(mp),'manifest_rows':len(m),'slot_rows':len(audit),
        'technical_slot_pass':int(audit.technical_pass.sum()),'qa_slot_pass':int(audit.qa_pass_recomputed.sum()),
        'records_all_three_technical':int(rec.all_three_technical_pass.sum()),
        'records_all_three_qa':int(rec.all_three_qa_pass_recomputed.sum()),
        'records_strict_t0_aligned_72h':int(rec.strict_t0_aligned_72h.sum()),
        'records_strict_model_ready':int(rec.strict_model_ready.sum()),
        'slots_with_errors':int(audit.error.fillna('').ne('').sum()),
        'slots_missing_time_metadata':int((~audit.time_metadata_present.fillna(False)).sum())
    }])
    summary.to_csv(summary_path,index=False)
    print('Manifest selected:',mp); print('\n',summary.to_string(index=False))
    print('\nTechnical failures:',int((~audit.technical_pass).sum()))
    if (~audit.technical_pass).any():
        print(audit.loc[~audit.technical_pass,['record_id','slot','band_count','height','width','finite_fraction','nonconstant_band_count','grid_matches_scl','scl_valid_fraction','center_offset_m','error']].head(30).to_string(index=False))
    print('\nSaved:')
    print(audit_path)
    print(rec_path)
    print(summary_path)

if __name__=='__main__': main()
