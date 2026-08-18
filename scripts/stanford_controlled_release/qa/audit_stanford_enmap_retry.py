#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
SEARCH = ROOT + "/search"
ITEMS = ROOT + "/collections/ENMAP_HSI_L2A/items"
COLLECTION = "ENMAP_HSI_L2A"
BBOX_HALF_DEG = 0.05

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="stanford_enmap_retry")
    p.add_argument("--window-min", type=int, default=90)
    p.add_argument("--exact-min", type=float, default=30)
    p.add_argument("--sleep", type=float, default=0.2)
    return p.parse_args()

def parse_dt(r):
    s = (r.get("datetime_UTC") or "").strip()
    if s:
        return datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)
    return datetime.fromisoformat(f"{r['date']}T{r['time_UTC']}+00:00").astimezone(timezone.utc)

def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def bbox(r):
    lat, lon = float(r["lat"]), float(r["lon"])
    h = BBOX_HALF_DEG
    return [lon-h, lat-h, lon+h, lat+h]

def get_json(session, url, params, tries=5):
    last = {"http_status":"", "error":"", "response_text":""}
    for i in range(1, tries+1):
        try:
            resp = session.get(url, params=params, timeout=90)
            last["http_status"] = resp.status_code
            last["response_text"] = resp.text[:1000].replace("\n"," ")
            if resp.status_code == 429:
                time.sleep(min(60, 2**i))
                continue
            resp.raise_for_status()
            return resp.json(), last
        except Exception as e:
            last["error"] = repr(e)
            if i < tries:
                time.sleep(min(20, 2**i))
    return None, last

def item_dt(item):
    p = item.get("properties") or {}
    s = p.get("datetime") or p.get("start_datetime")
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def assets(item):
    names, hrefs = [], []
    for k,v in (item.get("assets") or {}).items():
        names.append(str(k))
        if isinstance(v,dict) and v.get("href"):
            hrefs.append(str(v["href"]))
    return "|".join(names), "|".join(hrefs)

def write_csv(path, rows):
    rows = list(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        if not keys:
            return
        w = csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def query(session, r, start, end):
    bb = bbox(r)
    common = {
        "bbox": ",".join(map(str,bb)),
        "datetime": f"{iso(start)}/{iso(end)}",
        "limit": 100,
        "f": "application/geo+json",
    }

    p1 = dict(common)
    p1["collections"] = COLLECTION
    js, diag1 = get_json(session, SEARCH, p1)
    if js is not None:
        return js.get("features",[]) or [], "GET_SEARCH", diag1

    p2 = dict(common)
    js, diag2 = get_json(session, ITEMS, p2)
    if js is not None:
        return js.get("features",[]) or [], "COLLECTION_ITEMS", diag2

    d = {
        "http_status": f"search={diag1.get('http_status')} items={diag2.get('http_status')}",
        "error": f"search={diag1.get('error')} | items={diag2.get('error')}",
        "response_text": (
            f"search: {diag1.get('response_text','')} || "
            f"items: {diag2.get('response_text','')}"
        )[:1800]
    }
    return None, "BOTH_FAILED", d

def main():
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    with open(a.input,newline="",encoding="utf-8-sig") as f:
        allrows = list(csv.DictReader(f))

    rows = [r for r in allrows if (r.get("SatellitePlotName") or "").strip()=="EnMAP"]
    print("EnMAP events:", len(rows))

    s = requests.Session()
    s.headers.update({
        "User-Agent":"Stanford-EnMAP-Retry/2.0",
        "Accept":"application/geo+json, application/json;q=0.9, */*;q=0.1",
    })

    summaries, candidates, errors = [], [], []

    for i,r in enumerate(rows,1):
        rid = r.get("release_ID","")
        expected = parse_dt(r)
        print(f"[{i}/{len(rows)}] {rid}")

        st = expected - timedelta(minutes=a.window_min)
        en = expected + timedelta(minutes=a.window_min)
        feats, method, diag = query(s,r,st,en)
        mode = "PRIMARY_WINDOW"

        if feats == []:
            day0 = expected.replace(hour=0,minute=0,second=0,microsecond=0)
            feats, method, diag = query(s,r,day0,day0+timedelta(days=1))
            mode = "WHOLE_DAY_FALLBACK"

        if feats is None:
            status = "REQUEST_ERROR"
            errors.append({
                "release_ID":rid,
                "method":method,
                "http_status":diag.get("http_status",""),
                "error":diag.get("error",""),
                "response_text":diag.get("response_text",""),
            })
            feats = []
        else:
            status = "NOT_FOUND"

        cs = []
        for item in feats:
            dt = item_dt(item)
            d = None if dt is None else abs((dt-expected).total_seconds())/60.0
            names, hrefs = assets(item)
            p = item.get("properties") or {}
            x = {
                **r,
                "candidate_id":item.get("id",""),
                "candidate_datetime_utc":iso(dt) if dt else "",
                "abs_time_delta_min":d if d is not None else "",
                "eo_cloud_cover":p.get("eo:cloud_cover",""),
                "enmap_overall_quality":p.get("enmap:overallQuality",""),
                "asset_names":names,
                "asset_hrefs":hrefs,
                "query_method":method,
                "search_mode":mode,
            }
            cs.append(x)

        cs.sort(key=lambda x: float(x["abs_time_delta_min"]) if x["abs_time_delta_min"] != "" else 1e18)
        candidates.extend(cs)

        if cs:
            best = cs[0]
            d = float(best["abs_time_delta_min"]) if best["abs_time_delta_min"] != "" else None
            if d is None:
                status = "FOUND_TIME_UNKNOWN"
            elif d <= a.exact_min:
                status = "RESOLVED_EXACT"
            elif d <= a.window_min:
                status = "RESOLVED_NEARBY"
            else:
                status = "DATE_ONLY_REVIEW"
        else:
            best = {}

        summaries.append({
            **r,
            "availability_status":status,
            "catalog_candidate_count":len(cs),
            "best_candidate_id":best.get("candidate_id",""),
            "best_candidate_datetime_utc":best.get("candidate_datetime_utc",""),
            "best_abs_time_delta_min":best.get("abs_time_delta_min",""),
            "best_cloud_cover":best.get("eo_cloud_cover",""),
            "best_overall_quality":best.get("enmap_overall_quality",""),
            "best_asset_names":best.get("asset_names",""),
            "best_asset_hrefs":best.get("asset_hrefs",""),
            "query_method":method,
            "search_mode":mode,
            "http_status":diag.get("http_status",""),
            "request_error":diag.get("error",""),
        })
        print(" ", status, best.get("candidate_id",""))
        time.sleep(a.sleep)

    write_csv(out/"01_enmap_event_summary.csv", summaries)
    write_csv(out/"02_enmap_candidates.csv", candidates)
    write_csv(out/"03_request_errors.csv", errors)

    c = Counter(x["availability_status"] for x in summaries)
    with open(out/"SUMMARY.txt","w",encoding="utf-8") as f:
        f.write("Stanford 2025 EnMAP retry audit\n")
        f.write("="*60+"\n")
        f.write(f"Events: {len(rows)}\n")
        for k,v in sorted(c.items()):
            f.write(f"{k:28s}: {v}\n")
        f.write(f"Request-error rows: {len(errors)}\n")

    print()
    print((out/"SUMMARY.txt").read_text())

if __name__=="__main__":
    main()
