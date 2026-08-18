#!/usr/bin/env python3
from __future__ import annotations
import argparse, http.client, json, math, os, socket, ssl, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

STAC_ROOT="https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION_L2A="ENMAP_HSI_L2A"
ENMAP_ARCHIVE_START=pd.Timestamp("2022-04-27T00:00:00Z")
USER_AGENT="UAlberta-Methane-EnMAP-Audit/1.1"

def parse_mixed_utc(series):
    """Parse mixed datetime strings robustly across pandas versions."""
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        return series.apply(lambda x: pd.to_datetime(x, errors="coerce", utc=True))

def patch_carbon_mapper_from_recovery(df, cm_path):
    if not cm_path.exists():
        raise SystemExit(f"Missing Carbon Mapper recovery file: {cm_path}")

    cm = pd.read_csv(cm_path, low_memory=False)
    if "scene_timestamp_final" not in cm.columns:
        raise SystemExit("scene_timestamp_final missing from Carbon Mapper recovery file")

    cm_final = parse_mixed_utc(cm["scene_timestamp_final"])

    mask = df["dataset"].eq("CARBON_MAPPER_CH4_PLUMES")
    patched = 0

    for idx in df.index[mask]:
        sr = pd.to_numeric(df.at[idx, "source_row"], errors="coerce")
        if pd.isna(sr):
            continue
        sr = int(sr)
        if 0 <= sr < len(cm) and pd.notna(cm_final.iloc[sr]):
            if pd.isna(df.at[idx, "event_time_utc"]):
                patched += 1
            df.at[idx, "event_time_utc"] = cm_final.iloc[sr]

    return df, patched, int(cm_final.notna().sum())


def get_json(url,retries=8,timeout=120):
    retryable=(HTTPError,URLError,TimeoutError,ConnectionError,ConnectionResetError,BrokenPipeError,
               socket.timeout,ssl.SSLError,http.client.IncompleteRead,http.client.RemoteDisconnected,
               http.client.BadStatusLine,json.JSONDecodeError)
    last=None
    for attempt in range(1,retries+1):
        try:
            req=Request(url,headers={"User-Agent":USER_AGENT,"Accept":"application/geo+json, application/json","Connection":"close"})
            with urlopen(req,timeout=timeout) as r:
                payload=r.read()
            return json.loads(payload.decode("utf-8"))
        except retryable as e:
            last=e
            if attempt>=retries: break
            wait=min(45,2**(attempt-1))
            print(f"    targeted STAC retry {attempt}/{retries}: {type(e).__name__}: {e}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Targeted STAC query failed after {retries} attempts: {type(last).__name__}: {last}")

def point_on_segment(px,py,ax,ay,bx,by,eps=1e-10):
    cross=(px-ax)*(by-ay)-(py-ay)*(bx-ax)
    if abs(cross)>eps: return False
    dot=(px-ax)*(px-bx)+(py-ay)*(py-by)
    return dot<=eps

def point_in_ring(lon,lat,ring):
    if not ring or len(ring)<3: return False
    inside=False; j=len(ring)-1
    for i in range(len(ring)):
        xi,yi=ring[i][0],ring[i][1]; xj,yj=ring[j][0],ring[j][1]
        if point_on_segment(lon,lat,xi,yi,xj,yj): return True
        if (yi>lat)!=(yj>lat):
            denom=(yj-yi) if (yj-yi)!=0 else 1e-30
            if lon < (xj-xi)*(lat-yi)/denom+xi: inside=not inside
        j=i
    return inside

def point_in_polygon(lon,lat,coords):
    if not coords or not point_in_ring(lon,lat,coords[0]): return False
    return not any(point_in_ring(lon,lat,hole) for hole in coords[1:])

def point_in_geometry(lon,lat,geom):
    if not geom: return False
    typ=geom.get("type"); coords=geom.get("coordinates")
    if typ=="Polygon": return point_in_polygon(lon,lat,coords or [])
    if typ=="MultiPolygon": return any(point_in_polygon(lon,lat,p) for p in (coords or []))
    return False

def bbox_from_geometry(geom):
    if not geom: return None
    xs=[]; ys=[]
    def walk(x):
        if isinstance(x,(list,tuple)):
            if len(x)>=2 and isinstance(x[0],(int,float)) and isinstance(x[1],(int,float)):
                xs.append(float(x[0])); ys.append(float(x[1]))
            else:
                for y in x: walk(y)
    walk(geom.get("coordinates"))
    return [min(xs),min(ys),max(xs),max(ys)] if xs else None

def parse_dt(x):
    if x is None or x=="": return None
    t=pd.to_datetime(x,errors="coerce",utc=True)
    return None if pd.isna(t) else t

def normalize_feature(feat,collection):
    p=feat.get("properties",{}) or {}; geom=feat.get("geometry"); bbox=feat.get("bbox")
    if not bbox or len(bbox)<4: bbox=bbox_from_geometry(geom)
    dt=parse_dt(p.get("datetime")) or parse_dt(p.get("start_datetime")) or parse_dt(p.get("end_datetime"))
    datatake=p.get("enmap:datatakeID") or p.get("enmap:datatakeId") or p.get("enmap:datatake_id") or p.get("datatakeID")
    return {"id":feat.get("id"),"collection":collection,"datetime":dt,"geometry":geom,"bbox":bbox,
            "datatake_id":None if datatake is None else str(datatake),
            "overall_quality":p.get("enmap:overallQuality"),"cloud_cover":p.get("eo:cloud_cover"),"snow_cover":p.get("eo:snow_cover")}

def load_jsonl(path,collection):
    scenes=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                s=normalize_feature(json.loads(line),collection)
                if s["geometry"] and s["bbox"]: scenes.append(s)
            except Exception: pass
    return scenes

def grid_cell(lon,lat,cell_deg=1.0):
    return (math.floor((float(lon)+180.0)/cell_deg),math.floor((float(lat)+90.0)/cell_deg))

def build_grid_index(scenes,cell_deg=1.0):
    idx=defaultdict(list)
    for i,s in enumerate(scenes):
        b=s.get("bbox")
        if not b or len(b)<4: continue
        minx,miny,maxx,maxy=map(float,b[:4])
        x0,y0=grid_cell(minx,miny,cell_deg); x1,y1=grid_cell(maxx,maxy,cell_deg)
        for gx in range(x0,x1+1):
            for gy in range(y0,y1+1): idx[(gx,gy)].append(i)
    return idx

def spatial_matches(lon,lat,scenes,idx,cell_deg=1.0):
    out=[]
    for i in idx.get(grid_cell(lon,lat,cell_deg),[]):
        s=scenes[i]; b=s["bbox"]
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3] and point_in_geometry(lon,lat,s["geometry"]): out.append(i)
    return out

def choose_nearest(event_time,hit_ids,scenes):
    best_i=best_delta=best_abs=None
    for i in hit_ids:
        dt=scenes[i].get("datetime")
        if dt is None: continue
        delta=(dt-event_time).total_seconds()/3600.0; a=abs(delta)
        if best_abs is None or a<best_abs: best_i,best_delta,best_abs=i,delta,a
    return best_i,best_delta

def same_acquisition_candidates(l0_scene,lon,lat,l2_scenes,l2_idx,cell_deg=1.0):
    same=[]
    for i in spatial_matches(lon,lat,l2_scenes,l2_idx,cell_deg):
        s=l2_scenes[i]
        if l0_scene.get("datatake_id") and s.get("datatake_id") and l0_scene["datatake_id"]==s["datatake_id"]:
            same.append(i); continue
        if l0_scene.get("datetime") is not None and s.get("datetime") is not None:
            if abs((s["datetime"]-l0_scene["datetime"]).total_seconds())<=900: same.append(i)
    return same

def quality_name(x):
    try: return {0:"NOMINAL",1:"REDUCED",2:"LOW"}.get(int(x),str(x))
    except Exception: return None if x is None else str(x)

def load_targeted_cache(path):
    cache={}
    if not path.exists(): return cache
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            try: obj=json.loads(line)
            except Exception: continue
            if obj.get("status") in {"OK","EMPTY"}: cache[obj["key"]]=obj
    return cache

def targeted_query_for_l0_scene(l0_scene):
    b=l0_scene["bbox"]; dt=l0_scene["datetime"]
    if not b or dt is None: return None
    t0=dt-pd.Timedelta(minutes=15); t1=dt+pd.Timedelta(minutes=15)
    bbox_text=",".join(f"{float(x):.7f}" for x in b[:4])
    datetime_text=f"{t0.isoformat().replace('+00:00','Z')}/{t1.isoformat().replace('+00:00','Z')}"
    params=urlencode({"bbox":bbox_text,"datetime":datetime_text,"limit":100,"f":"json"})
    return f"{STAC_ROOT}collections/{COLLECTION_L2A}/items?{params}"

def fetch_targeted_l2a(l0_scene,cache,cache_path,error_log):
    key=str(l0_scene["id"])
    if key in cache: return cache[key]
    url=targeted_query_for_l0_scene(l0_scene)
    if url is None: return {"key":key,"status":"EMPTY","features":[]}
    try:
        doc=get_json(url); feats=doc.get("features",[]) or []
        obj={"key":key,"status":"OK" if feats else "EMPTY","url":url,"features":feats}
        with cache_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(obj,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())
        cache[key]=obj; return obj
    except Exception as e:
        with error_log.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"key":key,"url":url,"error_type":type(e).__name__,"error":str(e)},ensure_ascii=False)+"\n")
        return {"key":key,"status":"ERROR","features":[],"error":f"{type(e).__name__}: {e}"}

def temporal_class(abs_h):
    if abs_h<=24: return "MATCH_WITHIN_24H"
    if abs_h<=72: return "MATCH_WITHIN_72H"
    if abs_h<=24*7: return "MATCH_WITHIN_7D"
    if abs_h<=24*30: return "MATCH_WITHIN_30D"
    return "SPATIAL_COVERAGE_ONLY_GT30D"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",default=str(Path.home()/"methane_release_project/enmap_full_match_v1"))
    ap.add_argument("--cell-deg",type=float,default=1.0)
    args=ap.parse_args()
    work=Path(args.workdir).expanduser().resolve()
    project_root=work.parent
    master_path=project_root/"unified_enmap_input_v1/unified_enmap_input_deduped.csv"
    cm_recovery_path=project_root/"carbon_mapper_time_recovery_v1/carbon_mapper_all_CH4_plumes_time_recovered.csv"
    l0_path=work/"enmap_l0_stac_cache.jsonl"
    if not master_path.exists(): raise SystemExit(f"Missing original unified master: {master_path}")
    if not l0_path.exists(): raise SystemExit(f"Missing completed L0 cache: {l0_path}")

    l2_candidates=[work/"enmap_l2a_stac_cache.jsonl",work/"enmap_l2a_stac_cache.jsonl.part"]
    l2_partial_path=next((p for p in l2_candidates if p.exists() and p.stat().st_size>0),None)
    targeted_cache_path=work/"enmap_l2a_targeted_query_cache.jsonl"
    targeted_error_log=work/"enmap_l2a_targeted_query_errors.jsonl"
    checkpoint_path=work/"enmap_match_v4_checkpoint.csv"
    results_path=work/"enmap_match_results_v4.csv"
    summary_csv=work/"enmap_match_summary_by_dataset_v4.csv"
    summary_txt=work/"enmap_match_summary_v4.txt"

    print("="*84); print("ENMAP MATCH V4 — TIMESTAMP-REPAIRED TARGETED L2A"); print("="*84)
    df=pd.read_csv(master_path,low_memory=False)
    df["event_time_utc"]=parse_mixed_utc(df["event_time_utc"])

    before_by_dataset = (
        df.assign(_has_time=df["event_time_utc"].notna())
          .groupby("dataset")["_has_time"].sum()
          .astype(int)
          .to_dict()
    )

    df, cm_patched, cm_expected_full = patch_carbon_mapper_from_recovery(
        df, cm_recovery_path
    )

    after_by_dataset = (
        df.assign(_has_time=df["event_time_utc"].notna())
          .groupby("dataset")["_has_time"].sum()
          .astype(int)
          .to_dict()
    )

    corrected_master_path = work/"unified_enmap_input_timestamp_repaired_v4.csv"
    df.to_csv(corrected_master_path,index=False)

    print(f"Input records: {len(df)}")
    print(f"Carbon Mapper timestamps patched into unified master: {cm_patched}")
    print(f"Carbon Mapper recovery file full timestamps        : {cm_expected_full}")
    print()
    print("TIME COUNTS BY DATASET — BEFORE -> AFTER")
    for dataset in sorted(after_by_dataset):
        print(
            f"{dataset:36s} "
            f"{before_by_dataset.get(dataset,0):6d} -> {after_by_dataset.get(dataset,0):6d}"
        )

    # Hard validation: the previous V3 result proved that these two datasets
    # lost all timestamps during mixed-format parsing. Abort rather than
    # silently repeat that error.
    cm_after = after_by_dataset.get("CARBON_MAPPER_CH4_PLUMES", 0)
    ms_after = after_by_dataset.get("METHANESAT_POSNEG_222", 0)

    if cm_after < 32000:
        raise SystemExit(
            f"ABORT: Carbon Mapper timed rows only {cm_after}; expected about 32968."
        )
    if ms_after != 222:
        raise SystemExit(
            f"ABORT: MethaneSAT timed rows = {ms_after}; expected 222."
        )

    print("\nLoading completed global L0 cache...")
    l0_scenes=load_jsonl(l0_path,"ENMAP_HSI_L0_QL"); l0_idx=build_grid_index(l0_scenes,args.cell_deg)
    print(f"Usable L0 scenes: {len(l0_scenes)}")

    if l2_partial_path:
        print(f"\nLoading existing L2A metadata cache opportunistically:\n  {l2_partial_path}")
        l2_scenes=load_jsonl(l2_partial_path,COLLECTION_L2A)
        print(f"Usable already-cached L2A scenes: {len(l2_scenes)}")
    else:
        l2_scenes=[]; print("\nNo previous L2A cache found; targeted queries will fill needed matches.")
    l2_idx=build_grid_index(l2_scenes,args.cell_deg)
    targeted_cache=load_targeted_cache(targeted_cache_path)
    print(f"Previously completed targeted L2A queries: {len(targeted_cache)}")

    results=[]; targeted_query_count=0; targeted_query_errors=0
    for pos,(_,row) in enumerate(df.iterrows(),start=1):
        base=row.to_dict(); lat=pd.to_numeric(row.get("lat"),errors="coerce"); lon=pd.to_numeric(row.get("lon"),errors="coerce")
        event_time=row.get("event_time_utc")
        has_coords=pd.notna(lat) and pd.notna(lon) and -90<=float(lat)<=90 and -180<=float(lon)<=180
        has_time=pd.notna(event_time)
        out={"l0_spatial_scene_count":0,"l2a_same_acquisition_available":False,"l2a_source":None}
        if not has_coords:
            out["enmap_match_status"]="COORDS_MISSING"; base.update(out); results.append(base); continue
        lat=float(lat); lon=float(lon)
        l0_hits=spatial_matches(lon,lat,l0_scenes,l0_idx,args.cell_deg); out["l0_spatial_scene_count"]=len(l0_hits)
        if not has_time:
            out["enmap_match_status"]="TIME_MISSING_SPATIAL_COVERAGE" if l0_hits else "TIME_MISSING_NO_SPATIAL_COVERAGE"
            base.update(out); results.append(base); continue
        if event_time<ENMAP_ARCHIVE_START:
            out["enmap_match_status"]="PRE_ENMAP_ARCHIVE"; base.update(out); results.append(base); continue
        if not l0_hits:
            out["enmap_match_status"]="NO_ENMAP_SPATIAL_COVERAGE"; base.update(out); results.append(base); continue

        nearest_i,delta_h=choose_nearest(event_time,l0_hits,l0_scenes)
        if nearest_i is None:
            out["enmap_match_status"]="SPATIAL_COVERAGE_BUT_SCENE_TIME_MISSING"; base.update(out); results.append(base); continue

        l0=l0_scenes[nearest_i]; abs_h=abs(delta_h)
        out.update({"enmap_match_status":temporal_class(abs_h),"l0_nearest_scene_id":l0["id"],
                    "l0_nearest_time":l0["datetime"].isoformat() if l0["datetime"] is not None else None,
                    "l0_delta_hours_signed":delta_h,"l0_abs_delta_hours":abs_h,"l0_datatake_id":l0["datatake_id"],
                    "within_24h":abs_h<=24,"within_72h":abs_h<=72,"within_7d":abs_h<=24*7,"within_30d":abs_h<=24*30})

        same=same_acquisition_candidates(l0,lon,lat,l2_scenes,l2_idx,args.cell_deg)
        matched_l2=None
        if same:
            best_i,_=choose_nearest(l0["datetime"],same,l2_scenes)
            if best_i is not None:
                matched_l2=l2_scenes[best_i]; out["l2a_source"]="EXISTING_PARTIAL_OR_GLOBAL_CACHE"

        if matched_l2 is None:
            key=str(l0["id"]); was_cached=key in targeted_cache
            q=fetch_targeted_l2a(l0,targeted_cache,targeted_cache_path,targeted_error_log)
            if not was_cached: targeted_query_count+=1
            if q.get("status")=="ERROR":
                targeted_query_errors+=1; out["l2a_targeted_query_status"]="ERROR"; out["l2a_targeted_query_error"]=q.get("error")
            else:
                out["l2a_targeted_query_status"]=q.get("status")
                targeted_scenes=[]
                for feat in q.get("features",[]):
                    try:
                        s=normalize_feature(feat,COLLECTION_L2A)
                        if s["geometry"] and s["bbox"]: targeted_scenes.append(s)
                    except Exception: pass
                if targeted_scenes:
                    t_idx=build_grid_index(targeted_scenes,args.cell_deg)
                    same_t=same_acquisition_candidates(l0,lon,lat,targeted_scenes,t_idx,args.cell_deg)
                    if same_t:
                        best_i,_=choose_nearest(l0["datetime"],same_t,targeted_scenes)
                        if best_i is not None:
                            matched_l2=targeted_scenes[best_i]; out["l2a_source"]="TARGETED_STAC_QUERY"

        if matched_l2 is not None:
            out.update({"l2a_same_acquisition_available":True,"l2a_scene_id":matched_l2["id"],
                        "l2a_time":matched_l2["datetime"].isoformat() if matched_l2["datetime"] is not None else None,
                        "l2a_datatake_id":matched_l2["datatake_id"],"l2a_overall_quality_code":matched_l2["overall_quality"],
                        "l2a_overall_quality":quality_name(matched_l2["overall_quality"]),
                        "l2a_cloud_cover":matched_l2["cloud_cover"],"l2a_snow_cover":matched_l2["snow_cover"]})
        base.update(out); results.append(base)
        if pos%1000==0:
            print(f"[{pos}/{len(df)}] targeted new queries={targeted_query_count}, query errors={targeted_query_errors}")
        if pos%5000==0:
            pd.DataFrame(results).to_csv(checkpoint_path,index=False); print(f"  checkpoint -> {checkpoint_path}")

    res=pd.DataFrame(results); res.to_csv(results_path,index=False)
    for c in ["within_24h","within_72h","within_7d","within_30d","l2a_same_acquisition_available"]:
        if c not in res.columns: res[c]=False
        res[c]=res[c].fillna(False).astype(bool)

    rows=[]
    for dataset,g in res.groupby("dataset",dropna=False):
        rows.append({"dataset":dataset,"records":len(g),"within_24h":int(g["within_24h"].sum()),
                     "within_72h":int(g["within_72h"].sum()),"within_7d":int(g["within_7d"].sum()),
                     "within_30d":int(g["within_30d"].sum()),
                     "l2a_same_acquisition":int(g["l2a_same_acquisition_available"].sum()),
                     "targeted_query_errors":int((g["l2a_targeted_query_status"]=="ERROR").sum()) if "l2a_targeted_query_status" in g else 0})
    by_dataset=pd.DataFrame(rows); by_dataset.to_csv(summary_csv,index=False)
    status_counts=res["enmap_match_status"].value_counts(dropna=False)
    lines=["ENMAP FULL MATCH V4 SUMMARY","="*84,f"Input records                  : {len(res)}",
           f"Within 24 h                    : {int(res['within_24h'].sum())}",
           f"Within 72 h                    : {int(res['within_72h'].sum())}",
           f"Within 7 days                  : {int(res['within_7d'].sum())}",
           f"Within 30 days                 : {int(res['within_30d'].sum())}",
           f"L2A same acquisition available : {int(res['l2a_same_acquisition_available'].sum())}",
           f"Rows with exact/full time        : {int(parse_mixed_utc(res['event_time_utc']).notna().sum())}",
           f"Time-missing rows                : {int(parse_mixed_utc(res['event_time_utc']).isna().sum())}",
           f"New targeted L2A queries       : {targeted_query_count}",
           f"Targeted query errors          : {targeted_query_errors}","","MATCH STATUS COUNTS"]
    for k,v in status_counts.items(): lines.append(f"{str(k):42s} {int(v)}")
    lines+=["","BY DATASET",by_dataset.to_string(index=False),"","FILES",f"Results : {results_path}",
            f"Summary : {summary_csv}",f"Targeted L2A cache : {targeted_cache_path}",
            f"Targeted errors    : {targeted_error_log}",f"Corrected master   : {corrected_master_path}","","NOTE",
            "The incomplete global L2A cache was used only opportunistically.",
            "Missing L2A evidence was checked with small bbox+time STAC requests.",
            "A targeted server error does not abort the full methane audit."]
    summary_txt.write_text("\n".join(lines),encoding="utf-8"); print("\n"+"\n".join(lines))

if __name__=="__main__":
    main()
