#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess, math, re, time, traceback
import numpy as np
import pandas as pd
import ee

# ============================================================
# CONFIG
# ============================================================
HOME = Path.home()
PROJECT = HOME / "methane_release_project"
INPUT = PROJECT / "candidate_negative_validation" / "pilot_10_positive_40_candidates_s2qa.csv"
OUTDIR = PROJECT / "candidate_negative_validation" / "methaneair_coverage_first_v5"
OUTDIR.mkdir(parents=True, exist_ok=True)

EE_PROJECT = "methane-release-gee"
MAIR_L3 = "projects/edf-methanesat-ee/assets/mair/L3concentration"
MAIR_L4 = "projects/edf-methanesat-ee/assets/mair/L4point"
S2 = "COPERNICUS/S2_SR_HARMONIZED"

MAIR_PUBLIC_START = pd.Timestamp("2025-07-09")
MAIR_PUBLIC_END_EXCL = pd.Timestamp("2025-10-11")
SEARCH_MIN_D = 1
SEARCH_MAX_D = 45

PATCH_HALF_M = 240.0
MAIR_SCALE = 10.2
MIN_MAIR_VALID = 0.80
BG_INNER_M = 800.0
BG_OUTER_M = 2000.0
MIN_BG_VALID = 0.30
L4_RADIUS_M = 5000.0

S2_HOURS = 72.0
S2_SCALE = 20
MIN_S2_CLEAR = 0.80
GEE_WORKERS = 3
MAX_RETRIES = 4

# ============================================================
# LOAD 10 PARENTS
# ============================================================
if not INPUT.exists():
    raise FileNotFoundError(INPUT)

pilot = pd.read_csv(INPUT, low_memory=False)
pilot["_date"] = pd.to_datetime(pilot["Date"], errors="coerce")
if pilot["_date"].isna().any():
    raise RuntimeError("Invalid Date in pilot input")

parents = []
for parent_num, g in pilot.groupby("Pilot Parent Number", sort=True):
    pos_dates = {
        (pd.Timestamp(r["_date"]).normalize() - pd.Timedelta(days=int(r["Resolved Offset Days"])))
        for _, r in g.iterrows()
    }
    if len(pos_dates) != 1:
        raise RuntimeError(f"Parent {parent_num}: inconsistent positive dates {sorted(pos_dates)}")
    r0 = g.iloc[0]
    parents.append({
        "Pilot Parent Number": int(parent_num),
        "Source Positive Record ID": r0["Source Positive Record ID"],
        "Site": r0["Site"],
        "Latitude": float(r0["Latitude"]),
        "Longitude": float(r0["Longitude"]),
        "Parent Positive Date": next(iter(pos_dates)),
    })
parents = pd.DataFrame(parents)

print("=" * 112)
print("METHANEAIR COVERAGE-FIRST V5")
print("=" * 112)
print("Parents:", len(parents))
print(parents[["Pilot Parent Number","Site","Parent Positive Date"]].to_string(index=False))

ee.Initialize(project=EE_PROJECT)
print("Earth Engine ready.")

# ============================================================
# HELPERS
# ============================================================
def sf(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def utc(x):
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT

def req_count(region, scale):
    d = (ee.Image.constant(1).rename("requested").reduceRegion(
        reducer=ee.Reducer.count(), geometry=region, scale=scale,
        bestEffort=True, maxPixels=5_000_000).getInfo())
    return float(d.get("requested") or 0)

def band_count_median(image, band, region, scale):
    b = image.select(band)
    c = b.reduceRegion(ee.Reducer.count(), region, scale, bestEffort=True, maxPixels=5_000_000).getInfo().get(band)
    m = b.reduceRegion(ee.Reducer.median(), region, scale, bestEffort=True, maxPixels=5_000_000).getInfo().get(band)
    return float(c or 0), sf(m)

def flight_mid(props):
    a = utc(props.get("time_coverage_start"))
    b = utc(props.get("time_coverage_end"))
    if pd.notna(a) and pd.notna(b): return a + (b-a)/2
    if pd.notna(a): return a
    t = props.get("system:time_start")
    if t is not None:
        try: return pd.to_datetime(t, unit="ms", utc=True)
        except Exception: pass
    return pd.NaT

# ============================================================
# METHANEAIR L3 QA + XCH4 CONTEXT
# ============================================================
def mair_stats(image, point):
    src = point.buffer(PATCH_HALF_M).bounds()
    bg = point.buffer(BG_OUTER_M).difference(point.buffer(BG_INNER_M))

    src_req = req_count(src, MAIR_SCALE)
    src_valid, src_med = band_count_median(image, "XCH4", src, MAIR_SCALE)
    bg_req = req_count(bg, MAIR_SCALE)
    bg_valid, bg_med = band_count_median(image, "XCH4", bg, MAIR_SCALE)

    src_frac = src_valid/src_req if src_req else np.nan
    bg_frac = bg_valid/bg_req if bg_req else np.nan
    delta = src_med-bg_med if (np.isfinite(src_med) and np.isfinite(bg_med) and bg_frac >= MIN_BG_VALID) else np.nan

    return {
        "MethaneAIR Source Valid Fraction": src_frac,
        "MethaneAIR Source Valid Pixels": src_valid,
        "MethaneAIR Source XCH4 Median ppb": src_med,
        "MethaneAIR Background Valid Fraction": bg_frac,
        "MethaneAIR Background XCH4 Median ppb": bg_med,
        "MethaneAIR Source Minus Background ppb": delta,
        "MethaneAIR Source Valid Pass": bool(np.isfinite(src_frac) and src_frac >= MIN_MAIR_VALID),
    }

# ============================================================
# SAME-FLIGHT L4
# ============================================================
def l4_stats(flight_id, point):
    if not flight_id:
        return {"MethaneAIR Same-Flight L4 Count": np.nan, "MethaneAIR Nearby L4 Count <=5km": np.nan,
                "MethaneAIR Nearest L4 Distance m": np.nan, "MethaneAIR Nearby L4 Plume IDs": ""}

    fc = ee.FeatureCollection(MAIR_L4).filter(ee.Filter.eq("flight_id", str(flight_id)))
    total = int(fc.size().getInfo())

    def add_dist(f):
        return f.set("_distance_m", f.geometry().distance(point, 1))

    nearby = fc.map(add_dist).filter(ee.Filter.lte("_distance_m", L4_RADIUS_M)).sort("_distance_m")
    n = int(nearby.size().getInfo())
    ids, dists = [], []
    if n:
        for f in nearby.limit(200).getInfo().get("features", []):
            p = f.get("properties", {})
            ids.append(str(p.get("plume_id")))
            if p.get("_distance_m") is not None: dists.append(float(p["_distance_m"]))

    return {
        "MethaneAIR Same-Flight L4 Count": total,
        "MethaneAIR Nearby L4 Count <=5km": n,
        "MethaneAIR Nearest L4 Distance m": min(dists) if dists else np.nan,
        "MethaneAIR Nearby L4 Plume IDs": " | ".join(ids),
    }

# ============================================================
# SENTINEL-2 QA + +/-72 H MATCH
# ============================================================
def s2_qa(image, point):
    region = point.buffer(PATCH_HALF_M).bounds()
    scl = image.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).rename("clear")
    cloud = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10)).rename("cloud")
    shadow = scl.eq(3).rename("shadow")
    snow = scl.eq(11).rename("snow")
    sums = clear.addBands(cloud).addBands(shadow).addBands(snow).reduceRegion(
        ee.Reducer.sum(), region, S2_SCALE, bestEffort=True, maxPixels=2_000_000).getInfo()
    valid = scl.reduceRegion(ee.Reducer.count(), region, S2_SCALE, bestEffort=True, maxPixels=2_000_000).getInfo().get("SCL")
    req = req_count(region, S2_SCALE)
    valid = float(valid or 0); clear_px = float(sums.get("clear") or 0)
    clear_req = clear_px/req if req else np.nan
    return {
        "S2 Valid SCL Fraction": valid/req if req else np.nan,
        "S2 Clear Over Requested Fraction": clear_req,
        "S2 Clear Among Valid Fraction": clear_px/valid if valid else np.nan,
        "S2 Cloud Over Requested Fraction": float(sums.get("cloud") or 0)/req if req else np.nan,
        "S2 Shadow Over Requested Fraction": float(sums.get("shadow") or 0)/req if req else np.nan,
        "S2 Snow Over Requested Fraction": float(sums.get("snow") or 0)/req if req else np.nan,
        "_pass": bool(np.isfinite(clear_req) and clear_req >= MIN_S2_CLEAR),
    }

def nearest_s2(point, mid, positive_date):
    if pd.isna(mid): return {"S2 Match Status": "NO_FLIGHT_TIME"}
    a = mid - pd.Timedelta(hours=S2_HOURS)
    b = mid + pd.Timedelta(hours=S2_HOURS)
    ic = ee.ImageCollection(S2).filterBounds(point.buffer(PATCH_HALF_M).bounds()).filterDate(
        a.strftime("%Y-%m-%dT%H:%M:%S"), (b+pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")).sort("system:time_start")
    n = int(ic.size().getInfo())
    if n == 0: return {"S2 Match Status":"NO_S2_WITHIN_72H","S2 Scenes Within 72h":0}

    rows = []
    arr = ic.toList(n)
    pos = pd.Timestamp(positive_date).normalize()
    for i in range(n):
        im = ee.Image(arr.get(i))
        p = im.toDictionary(["system:index","system:time_start","PRODUCT_ID","MGRS_TILE","CLOUDY_PIXEL_PERCENTAGE"]).getInfo()
        if p.get("system:time_start") is None: continue
        t = pd.to_datetime(p["system:time_start"], unit="ms", utc=True)
        if t.tz_convert(None).normalize() <= pos: continue
        q = s2_qa(im, point)
        rows.append({"time":t,"delta_h":(t-mid).total_seconds()/3600.0,"product":p.get("PRODUCT_ID"),"tile":p.get("MGRS_TILE"),"scene_cloud":p.get("CLOUDY_PIXEL_PERCENTAGE"),**q})

    if not rows: return {"S2 Match Status":"NO_POST_POSITIVE_S2_WITHIN_72H","S2 Scenes Within 72h":n}
    good = [r for r in rows if r["_pass"]]
    if not good:
        best = min(rows, key=lambda r:(abs(r["delta_h"]), -sf(r["S2 Clear Over Requested Fraction"])))
        return {"S2 Match Status":"S2_WITHIN_72H_BUT_QA_FAIL","S2 Scenes Within 72h":n,
                "S2 Best Fail Delta Hours":best["delta_h"],"S2 Best Fail Clear Fraction":best["S2 Clear Over Requested Fraction"]}

    best = min(good, key=lambda r:(abs(r["delta_h"]), -sf(r["S2 Clear Over Requested Fraction"])))
    out = {k:v for k,v in best.items() if k not in ["_pass","time","delta_h","product","tile","scene_cloud"]}
    out.update({"S2 Match Status":"MATCHED_QA_PASS","S2 Scenes Within 72h":n,"S2 Datetime UTC":str(best["time"]),
                "S2 Date":best["time"].strftime("%Y-%m-%d"),"S2 Delta Hours From MethaneAIR":best["delta_h"],
                "S2 Abs Delta Hours From MethaneAIR":abs(best["delta_h"]),"S2 Product ID":best["product"],"S2 MGRS Tile":best["tile"],
                "S2 Scene Cloud Percentage":best["scene_cloud"]})
    return out

# ============================================================
# GEE: ONE PARENT
# ============================================================
def gee_parent(parent):
    pn = int(parent["Pilot Parent Number"])
    pos = pd.Timestamp(parent["Parent Positive Date"]).normalize()
    point = ee.Geometry.Point([float(parent["Longitude"]), float(parent["Latitude"])])
    patch = point.buffer(PATCH_HALF_M).bounds()

    requested_start = pos + pd.Timedelta(days=SEARCH_MIN_D)
    requested_end = pos + pd.Timedelta(days=SEARCH_MAX_D+1)
    start = max(requested_start, MAIR_PUBLIC_START)
    end = min(requested_end, MAIR_PUBLIC_END_EXCL)

    base = {"Pilot Parent Number":pn,"Source Positive Record ID":parent["Source Positive Record ID"],"Site":parent["Site"],
            "Latitude":parent["Latitude"],"Longitude":parent["Longitude"],"Parent Positive Date":pos.strftime("%Y-%m-%d"),
            "Requested Search Start":requested_start.strftime("%Y-%m-%d"),"Requested Search End":(requested_end-pd.Timedelta(days=1)).strftime("%Y-%m-%d")}

    if start >= end:
        return {"summary":{**base,"L3 Intersecting Images":0,"L3 Source-Valid Flights":0,"B With Aligned S2":0,"Parent Result":"OUTSIDE_PUBLIC_METHANEAIR_WINDOW"},"flights":[]}

    last = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            ic = ee.ImageCollection(MAIR_L3).filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")).filterBounds(patch).sort("system:time_start")
            n = int(ic.size().getInfo()); arr = ic.toList(n); flights = []
            for i in range(n):
                im = ee.Image(arr.get(i))
                props = im.toDictionary(["system:index","system:time_start","flight_id","target_id","time_coverage_start","time_coverage_end","processing_id"]).getInfo()
                mid = flight_mid(props)
                stats = mair_stats(im, point)
                r = {**base,"MethaneAIR Flight ID":props.get("flight_id"),"MethaneAIR Target ID":props.get("target_id"),
                     "MethaneAIR Time Start UTC":props.get("time_coverage_start"),"MethaneAIR Time End UTC":props.get("time_coverage_end"),
                     "MethaneAIR Midpoint UTC":str(mid) if pd.notna(mid) else "","Actual Days After Positive":((mid.tz_convert(None).normalize()-pos).days if pd.notna(mid) else np.nan),**stats}

                if not stats["MethaneAIR Source Valid Pass"]:
                    r.update({"S2 Match Status":"NOT_SEARCHED_L3_COVERAGE_FAIL","Coverage-First Classification":"U_METHANEAIR_L3_PARTIAL_OR_INVALID_AT_SOURCE"})
                    flights.append(r); continue

                l4 = l4_stats(props.get("flight_id"), point); r.update(l4)
                s2 = nearest_s2(point, mid, pos); r.update(s2)
                near = l4["MethaneAIR Nearby L4 Count <=5km"]
                total = l4["MethaneAIR Same-Flight L4 Count"]
                if pd.notna(near) and int(near) > 0:
                    cls = "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM"
                elif pd.notna(total) and int(total) > 0:
                    cls = "B_HIGH_RES_NO_L4_DETECTION_WITH_ALIGNED_S2" if s2.get("S2 Match Status") == "MATCHED_QA_PASS" else "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
                else:
                    cls = "U_VALID_L3_AND_S2_BUT_L4_AVAILABILITY_UNCERTAIN" if s2.get("S2 Match Status") == "MATCHED_QA_PASS" else "U_VALID_L3_L4_AVAILABILITY_UNCERTAIN_NO_USABLE_S2"
                r["Coverage-First Classification"] = cls
                flights.append(r)

            # dedup by flight_id, keep highest source coverage
            if flights:
                f = pd.DataFrame(flights)
                f["_key"] = f["MethaneAIR Flight ID"].astype("string")
                f.loc[f["_key"].isna(),"_key"] = f.index.astype(str)
                f["_vf"] = pd.to_numeric(f["MethaneAIR Source Valid Fraction"],errors="coerce").fillna(-1)
                f = f.sort_values("_vf",ascending=False).drop_duplicates("_key").drop(columns=["_key","_vf"])
                flights = f.to_dict("records")

            valid = sum(bool(x.get("MethaneAIR Source Valid Pass")) for x in flights)
            bcount = sum(x.get("Coverage-First Classification") == "B_HIGH_RES_NO_L4_DETECTION_WITH_ALIGNED_S2" for x in flights)
            rejects = sum(x.get("Coverage-First Classification") == "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM" for x in flights)
            if bcount: result = "B_CANDIDATE_WITH_ALIGNED_S2_FOUND"
            elif rejects: result = "VALID_L3_FOUND_BUT_NEARBY_L4_REJECT"
            elif valid: result = "VALID_L3_FOUND_BUT_NO_COMPLETE_B_PLUS_S2"
            elif n: result = "L3_INTERSECTS_BUT_SOURCE_COVERAGE_INVALID"
            else: result = "NO_L3_FLIGHT_INTERSECTION"
            return {"summary":{**base,"L3 Intersecting Images":n,"L3 Source-Valid Flights":valid,"B With Aligned S2":bcount,"Nearby L4 Reject Flights":rejects,"Parent Result":result},"flights":flights}
        except Exception as e:
            last = e; time.sleep(2*attempt)

    return {"summary":{**base,"Parent Result":"QUERY_ERROR","Query Error":repr(last)},"flights":[]}

# ============================================================
# CACHE-ONLY MAC / LAB BRANCHES — NO RECURSIVE SCAN
# ============================================================
CACHE = {
    "LOCAL":[
        PROJECT/"candidate_negative_validation"/"methaneair_2025_parallel_v4"/"local_methaneair_file_inventory.txt",
        PROJECT/"candidate_negative_validation"/"actual_s2_45day_parallel_v3"/"04_local_metadata_inventory.txt",
        PROJECT/"candidate_negative_validation"/"parallel_multisource_40"/"local_existing_sensor_files.txt"],
    "LAB":[
        PROJECT/"candidate_negative_validation"/"methaneair_2025_parallel_v4"/"lab_methaneair_file_inventory.txt",
        PROJECT/"candidate_negative_validation"/"actual_s2_45day_parallel_v3"/"04_lab_metadata_inventory.txt",
        PROJECT/"candidate_negative_validation"/"parallel_multisource_40"/"lab_existing_sensor_files.txt"]}

def lab_mounted():
    try:
        return "/Volumes/engg-leung" in subprocess.run(["mount"],capture_output=True,text=True,timeout=20).stdout
    except Exception: return False

def load_cache(label):
    paths=[]; seen=set(); used=[]
    for f in CACHE[label]:
        if not f.exists(): continue
        used.append(str(f))
        for line in f.read_text(encoding="utf-8",errors="ignore").splitlines():
            line=line.strip()
            if not line or line.startswith("STATUS:") or line.startswith("COUNT:"): continue
            if line not in seen: seen.add(line); paths.append(Path(line))
    return paths,used

def norm(s): return str(s).strip().lower().replace(" ","_").replace("-","_")
def col(cols,names):
    m={norm(c):c for c in cols}
    for n in names:
        if norm(n) in m: return m[norm(n)]
    for n in names:
        for k,v in m.items():
            if norm(n) in k: return v
    return None

def hav(lat0,lon0,lats,lons):
    R=6371.0088; p0=math.radians(float(lat0)); q0=math.radians(float(lon0)); p=np.radians(np.asarray(lats,float)); q=np.radians(np.asarray(lons,float))
    a=np.sin((p-p0)/2)**2+math.cos(p0)*np.cos(p)*np.sin((q-q0)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cache_branch(label):
    paths,used=load_cache(label); mounted = lab_mounted() if label=="LAB" else True
    print(f"[{label}] cache files={len(used)} paths={len(paths)} mounted={mounted}")
    out=OUTDIR/f"{label.lower()}_cache_paths.txt"
    with out.open("w",encoding="utf-8") as f:
        for u in used: f.write(f"CACHE: {u}\n")
        f.write(f"PATHS: {len(paths)}\nMOUNTED: {mounted}\n\n")
        for p in paths: f.write(str(p)+"\n")

    # Strong content matching only for local CSVs; LAB remains cache-index-only to avoid SMB hangs.
    matches=[]
    if label=="LOCAL":
        for p in paths:
            if p.suffix.lower() != ".csv" or not p.exists(): continue
            try: df=pd.read_csv(p,low_memory=False)
            except Exception: continue
            la=col(df.columns,["latitude","lat","plume_latitude"]); lo=col(df.columns,["longitude","lon","lng","plume_longitude"])
            tc=col(df.columns,["time_coverage_start","datetime","timestamp","observation_time","acquisition_time","date"])
            if la is None or lo is None or tc is None: continue
            lats=pd.to_numeric(df[la],errors="coerce"); lons=pd.to_numeric(df[lo],errors="coerce"); times=pd.to_datetime(df[tc],errors="coerce",utc=True)
            ok=lats.notna()&lons.notna()&times.notna()
            if not ok.any(): continue
            tmp=pd.DataFrame({"lat":lats,"lon":lons,"time":times}).loc[ok].copy()
            for _,parent in parents.iterrows():
                pos=pd.Timestamp(parent["Parent Positive Date"]).tz_localize("UTC"); dd=(tmp["time"]-pos).dt.total_seconds()/86400.0
                sub=tmp[(dd>=SEARCH_MIN_D)&(dd<=SEARCH_MAX_D+1)].copy()
                if sub.empty: continue
                sub["dist"]=hav(parent["Latitude"],parent["Longitude"],sub["lat"],sub["lon"]); sub=sub[sub["dist"]<=10]
                for _,h in sub.iterrows():
                    matches.append({"Origin":label,"Pilot Parent Number":int(parent["Pilot Parent Number"]),"Site":parent["Site"],"File":str(p),"Metadata Datetime UTC":str(h["time"]),"Distance km":float(h["dist"])})
    return {"paths":len(paths),"used":used,"mounted":mounted,"matches":pd.DataFrame(matches)}

# ============================================================
# PARALLEL EXECUTION
# ============================================================
def gee_all():
    summaries=[]; flights=[]
    with ThreadPoolExecutor(max_workers=GEE_WORKERS) as pool:
        fm={pool.submit(gee_parent,p):int(p["Pilot Parent Number"]) for _,p in parents.iterrows()}
        done=0
        for fut in as_completed(fm):
            pn=fm[fut]
            try:
                r=fut.result(); summaries.append(r["summary"]); flights.extend(r["flights"])
            except Exception as e:
                summaries.append({"Pilot Parent Number":pn,"Parent Result":"BRANCH_ERROR","Query Error":repr(e)})
            done+=1; print(f"[GEE] parent {pn} complete ({done}/{len(parents)})")
    return pd.DataFrame(summaries),pd.DataFrame(flights)

print("\nStarting GEE + LOCAL cache + LAB cache concurrently...")
with ThreadPoolExecutor(max_workers=3) as pool:
    fm={pool.submit(gee_all):"GEE",pool.submit(cache_branch,"LOCAL"):"LOCAL",pool.submit(cache_branch,"LAB"):"LAB"}; results={}
    for fut in as_completed(fm):
        name=fm[fut]
        try: results[name]=fut.result(); print(f"[{name}] COMPLETE")
        except Exception as e: results[name]={"error":repr(e),"trace":traceback.format_exc()}; print(f"[{name}] FAILED {e!r}")

parent_summary, flights = results.get("GEE",(pd.DataFrame(),pd.DataFrame())) if isinstance(results.get("GEE"),tuple) else (pd.DataFrame(),pd.DataFrame())
local=results.get("LOCAL",{}); lab=results.get("LAB",{})
local_matches=local.get("matches",pd.DataFrame()) if isinstance(local,dict) else pd.DataFrame()
lab_matches=lab.get("matches",pd.DataFrame()) if isinstance(lab,dict) else pd.DataFrame()
cache_matches=pd.concat([x for x in [local_matches,lab_matches] if isinstance(x,pd.DataFrame) and len(x)],ignore_index=True,sort=False) if any(isinstance(x,pd.DataFrame) and len(x) for x in [local_matches,lab_matches]) else pd.DataFrame()

# ============================================================
# SAVE
# ============================================================
PARENT_OUT=OUTDIR/"01_parent_coverage_summary.csv"; FLIGHT_OUT=OUTDIR/"02_methaneair_flight_level_inventory.csv"; CACHE_OUT=OUTDIR/"03_local_lab_cached_metadata_matches.csv"
parent_summary.to_csv(PARENT_OUT,index=False,encoding="utf-8-sig"); flights.to_csv(FLIGHT_OUT,index=False,encoding="utf-8-sig"); cache_matches.to_csv(CACHE_OUT,index=False,encoding="utf-8-sig")

# one best row per parent
best=[]
priority={"B_HIGH_RES_NO_L4_DETECTION_WITH_ALIGNED_S2":1,"U_VALID_L3_AND_S2_BUT_L4_AVAILABILITY_UNCERTAIN":2,
          "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2":3,"U_VALID_L3_L4_AVAILABILITY_UNCERTAIN_NO_USABLE_S2":4,
          "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM":5,"U_METHANEAIR_L3_PARTIAL_OR_INVALID_AT_SOURCE":6}
for _,p in parents.iterrows():
    pn=int(p["Pilot Parent Number"]); base={"Pilot Parent Number":pn,"Source Positive Record ID":p["Source Positive Record ID"],"Site":p["Site"],"Latitude":p["Latitude"],"Longitude":p["Longitude"],"Parent Positive Date":pd.Timestamp(p["Parent Positive Date"]).strftime("%Y-%m-%d")}
    sub=flights[flights["Pilot Parent Number"].astype(int)==pn].copy() if not flights.empty else pd.DataFrame()
    if sub.empty: best.append({**base,"Best Candidate Status":"NO_METHANEAIR_FLIGHT_INTERSECTION"}); continue
    sub["_p"] = sub["Coverage-First Classification"].map(priority).fillna(99)
    if "S2 Abs Delta Hours From MethaneAIR" in sub.columns:
        sub["_dt"] = pd.to_numeric(sub["S2 Abs Delta Hours From MethaneAIR"], errors="coerce").fillna(9999)
    else:
        sub["_dt"] = 9999.0
    sub["_vf"] = pd.to_numeric(sub["MethaneAIR Source Valid Fraction"], errors="coerce").fillna(-1)
    r=sub.sort_values(["_p","_dt","_vf"],ascending=[True,True,False]).iloc[0].drop(labels=["_p","_dt","_vf"],errors="ignore").to_dict(); r["Best Candidate Status"]=r.get("Coverage-First Classification"); best.append(r)
best=pd.DataFrame(best); BEST_CSV=OUTDIR/"04_best_candidate_per_parent.csv"; BEST_XLSX=OUTDIR/"04_best_candidate_per_parent.xlsx"; best.to_csv(BEST_CSV,index=False,encoding="utf-8-sig"); best.to_excel(BEST_XLSX,index=False,engine="openpyxl")

# ============================================================
# SUMMARY
# ============================================================
print("\n"+"="*112); print("V5 COVERAGE-FIRST SUMMARY"); print("="*112)
print("\nParent result:"); print(parent_summary["Parent Result"].value_counts(dropna=False) if "Parent Result" in parent_summary else "No parent summary")
if not flights.empty:
    vf=pd.to_numeric(flights["MethaneAIR Source Valid Fraction"],errors="coerce")
    print(f"\nFlight records: {len(flights)}")
    print(f"Source-valid flights >=80%: {int((vf>=MIN_MAIR_VALID).sum())}")
    print("\nCoverage-first classification:"); print(flights["Coverage-First Classification"].value_counts(dropna=False))
    valid=flights[vf>=MIN_MAIR_VALID]
    print("\nS2 matching among source-valid flights:"); print(valid["S2 Match Status"].value_counts(dropna=False) if len(valid) else "No source-valid flights")
print("\nBest candidate per parent:"); print(best["Best Candidate Status"].value_counts(dropna=False))
print(f"\nLOCAL cached paths: {local.get('paths',0) if isinstance(local,dict) else 0}")
print(f"LAB cached paths: {lab.get('paths',0) if isinstance(lab,dict) else 0} | mounted={lab.get('mounted') if isinstance(lab,dict) else None}")
print("\nOUTPUTS:"); print(PARENT_OUT); print(FLIGHT_OUT); print(CACHE_OUT); print(BEST_CSV); print(BEST_XLSX)
print("\n✅ MethaneAIR valid coverage first")
print("✅ all 10 parents in one batch")
print("✅ +1..+45 d MethaneAIR flights")
print("✅ same-flight L4 within 5 km")
print("✅ local-vs-background XCH4 context")
print("✅ Sentinel-2 nearest QA-pass within +/-72 h")
print("✅ Mac/Lab existing caches read concurrently")
print("✅ NO recursive Lab SMB scan")
print("✅ NO imagery downloaded")
