#!/usr/bin/env python3
"""
Build REAL temporal EMIT frames for the final MethaneFuse handoff.

This script reuses the already-tested v4 EMIT->MethaneFuse adapter for:
  - RFL-only Earthdata download
  - 16-band WV3 simulation
  - 480 m crop -> 16x518x518 TIFF
  - raw-L2A cleanup

Workflow
========
1) SEARCH (metadata only; no large downloads)
   Find real same-location EMITL2ARFL observations near t0-90 and t0-180.

2) BUILD
   Reuse the validated t0 TIFF, download/build real t-90 and t-180 frames,
   and keep only complete POS/NEG pairs after pixel QA.

MethaneFuse compatibility
=========================
The current generic MethaneFuse wide-table loader requires columns named:
  emit_0_path, emit_90_path, emit_360_path
while MethaneUnion's EMIT preprocessing uses 0/-90/-180.
Therefore:
  emit_0_path   -> real t0
  emit_90_path  -> real ~t-90
  emit_360_path -> real ~t-180   (compatibility alias; metadata records truth)

Coverage tiers
==============
strict30   : each target within +/-30 days
expanded60 : each target within +/-60 days
all120     : each target within +/-120 days
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TIER_LIMIT = {"strict30": 30.0, "expanded60": 60.0, "all120": 120.0}
TIER_RANK = {"none": 0, "all120": 1, "expanded60": 2, "strict30": 3}
REQ_RANK = {"all120": 1, "expanded60": 2, "strict30": 3}
MAX_BUILD_CANDIDATES = 12


def load_v4(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing v4 adapter: {path}\n"
            "Put prepare_emit_for_methanefuse_v4_resume_rflonly.py in the project root."
        )
    spec = importlib.util.spec_from_file_location("emit_v4", path)
    mod = importlib.util.module_from_spec(spec)
    # Register the dynamically loaded module before executing it.
    # Python dataclasses looks up cls.__module__ in sys.modules.
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_dt(value: str) -> datetime:
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def login(v4):
    try:
        v4.earthaccess.login(strategy="netrc")
        print("Authenticated using ~/.netrc")
    except Exception:
        v4.earthaccess.login(strategy="interactive", persist=False)
        print("Authenticated interactively for this run")


def candidates(v4, lon: float, lat: float, t0: datetime, target_days: int, limit_days: float):
    target = t0 - timedelta(days=target_days)
    start = target - timedelta(days=limit_days)
    end = min(target + timedelta(days=limit_days), t0 - timedelta(hours=12))
    if end <= start:
        return []

    hits = v4.earthaccess.search_data(
        short_name="EMITL2ARFL",
        version="001",
        point=(lon, lat),
        temporal=(start.isoformat(), end.isoformat()),
        count=1000,
    )

    uniq = {}
    for g in hits:
        dt = v4.granule_datetime(g)
        if dt is None or dt >= t0 - timedelta(hours=12):
            continue
        actual = (t0 - dt).total_seconds() / 86400.0
        error = abs(actual - target_days)
        if error > limit_days:
            continue
        ur = v4.granule_ur(g)
        rec = (error, actual, dt, ur, g)
        if ur not in uniq or rec[:2] < uniq[ur][:2]:
            uniq[ur] = rec
    return sorted(uniq.values(), key=lambda x: (x[0], x[1], x[3]))


def choose_distinct(v4, lon: float, lat: float, t0: datetime, limit_days: float):
    c90 = candidates(v4, lon, lat, t0, 90, limit_days)
    c180 = candidates(v4, lon, lat, t0, 180, limit_days)
    best = None
    for a in c90[:40]:
        for b in c180[:40]:
            if a[3] == b[3]:
                continue
            score = (a[0] + b[0], max(a[0], b[0]), a[0], b[0])
            if best is None or score < best[0]:
                best = (score, a, b)
    return (best[1], best[2], c90, c180) if best else (None, None, c90, c180)


def tier_from_errors(e90: float, e180: float):
    m = max(e90, e180)
    if m <= 30:
        return "strict30"
    if m <= 60:
        return "expanded60"
    if m <= 120:
        return "all120"
    return "none"


def cmd_search(a):
    root = Path(a.input_root).expanduser().resolve()
    src = root / "eval_emit_480m.csv"
    out = Path(a.search_out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    v4 = load_v4(Path(a.adapter_v4))
    rows = read_csv(src)

    print(f"Input rows: {len(rows)}")
    login(v4)
    result = []

    for i, r in enumerate(rows, 1):
        sid = r["id"]
        t0 = parse_dt(r["source_t0_utc"])
        lat, lon = float(r["latitude"]), float(r["longitude"])
        print(f"[{i:03d}/{len(rows):03d}] {sid} t0={t0.date()}")

        s90, s180, c90, c180 = choose_distinct(v4, lon, lat, t0, 120.0)
        rec = {
            "id": sid,
            "pair_id": r["pair_id"],
            "label": int(r["label"]),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "source_t0_utc": r["source_t0_utc"],
            "source_ch4enh_scene_id": r.get("source_ch4enh_scene_id", ""),
            "candidate_count_t90": len(c90),
            "candidate_count_t180": len(c180),
        }
        if s90 is None or s180 is None:
            rec.update({"search_status": "FAIL", "coverage_tier": "none"})
            print(f"  -> FAIL: t90 candidates={len(c90)} t180 candidates={len(c180)}")
        else:
            e90, o90, dt90, ur90, _ = s90
            e180, o180, dt180, ur180, _ = s180
            tier = tier_from_errors(e90, e180)
            rec.update({
                "search_status": "PASS",
                "coverage_tier": tier,
                "t90_granule": ur90,
                "t90_utc": dt90.isoformat(),
                "t90_actual_days_before_t0": f"{o90:.3f}",
                "t90_target_error_days": f"{e90:.3f}",
                "t180_granule": ur180,
                "t180_utc": dt180.isoformat(),
                "t180_actual_days_before_t0": f"{o180:.3f}",
                "t180_target_error_days": f"{e180:.3f}",
            })
            print(f"  -> {tier}: t90={o90:.1f}d (err {e90:.1f}), t180={o180:.1f}d (err {e180:.1f})")
        result.append(rec)
        write_csv(out / "temporal_selection.partial.csv", result)

    write_csv(out / "temporal_selection.csv", result)

    pair_rows = []
    for pid in sorted({r["pair_id"] for r in result}):
        g = [r for r in result if r["pair_id"] == pid]
        labels = {int(r["label"]) for r in g}
        pr = {"pair_id": pid, "samples": len(g), "has_pos_neg": int(labels == {0, 1})}
        for tier in ("strict30", "expanded60", "all120"):
            pr[f"pair_ready_{tier}"] = int(
                len(g) == 2
                and labels == {0, 1}
                and all(TIER_RANK.get(r.get("coverage_tier", "none"), 0) >= REQ_RANK[tier] for r in g)
            )
        pair_rows.append(pr)
    write_csv(out / "temporal_pair_coverage.csv", pair_rows)

    print("\n" + "=" * 72)
    print("SEARCH COMPLETE")
    print("=" * 72)
    for tier in ("strict30", "expanded60", "all120"):
        sample_n = sum(TIER_RANK.get(r.get("coverage_tier", "none"), 0) >= REQ_RANK[tier] for r in result)
        pair_n = sum(int(r[f"pair_ready_{tier}"]) for r in pair_rows)
        print(f"{tier:10s}: sample-ready={sample_n:3d}  complete pairs={pair_n:3d}")
    print(f"Selection: {out/'temporal_selection.csv'}")
    print(f"Pairs:     {out/'temporal_pair_coverage.csv'}")


def valid_tif(v4, p: Path):
    if not p.exists():
        return False
    try:
        with v4.rasterio.open(p) as ds:
            return (ds.count, ds.height, ds.width) == (16, 518, 518)
    except Exception:
        return False


def build_frame(v4, lon, lat, t0, target_days, limit_days, exclude_urs, srf_csv, raw, out_tif):
    cands = [x for x in candidates(v4, lon, lat, t0, target_days, limit_days) if x[3] not in exclude_urs]
    errs = []
    for i, (err, actual, dt, ur, g) in enumerate(cands[:MAX_BUILD_CANDIDATES], 1):
        print(f"     try t-{target_days} #{i}: actual={actual:.1f}d err={err:.1f}d")
        try:
            v4.cleanup_raw_l2a_cache(raw)
            nc = v4.download_l2a(g, raw)
            qa = v4.make_wv3_query_crop(nc, lon, lat, srf_csv, out_tif)
            v4.cleanup_raw_l2a_cache(raw)
            return {
                "granule": ur,
                "utc": dt.isoformat(),
                "actual": actual,
                "error": err,
                "missing": qa.get("native_missing_ratio", ""),
            }
        except Exception as exc:
            errs.append(f"{ur}:{exc}")
            v4.cleanup_raw_l2a_cache(raw)
            try:
                out_tif.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError(f"no_valid_tminus{target_days}: " + " | ".join(errs[-4:]))


def cmd_build(a):
    root = Path(a.input_root).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    v4 = load_v4(Path(a.adapter_v4))
    selection = read_csv(Path(a.search_csv).expanduser().resolve())
    req, limit_days = REQ_RANK[a.tier], TIER_LIMIT[a.tier]

    ready = [r for r in selection if TIER_RANK.get(r.get("coverage_tier", "none"), 0) >= req]
    complete = set()
    for pid in {r["pair_id"] for r in ready}:
        g = [r for r in ready if r["pair_id"] == pid]
        if len(g) == 2 and {int(r["label"]) for r in g} == {0, 1}:
            complete.add(pid)
    work = sorted([r for r in ready if r["pair_id"] in complete], key=lambda r: (r["pair_id"], -int(r["label"])))

    print(f"Tier: {a.tier} (each target within +/-{limit_days:.0f}d)")
    print(f"Search-ready complete pairs: {len(complete)}")
    print(f"Samples queued: {len(work)}")
    if not work:
        raise RuntimeError("No complete temporal pairs for this tier")

    srf = out / "WV3_VNIR_SWIR_response.csv"
    shutil.copy2(root / "WV3_VNIR_SWIR_response.csv", srf)
    if (root / "make_paths_absolute.py").exists():
        shutil.copy2(root / "make_paths_absolute.py", out / "make_paths_absolute.py")
    raw = out / "raw_l2a"
    raw.mkdir(exist_ok=True)
    login(v4)

    progress_path = out / "temporal_build_progress.csv"
    previous = read_csv(progress_path) if progress_path.exists() and progress_path.stat().st_size else []
    prev = {r["id"]: r for r in previous}
    progress, built = [], []

    for i, r in enumerate(work, 1):
        sid, pid, label = r["id"], r["pair_id"], int(r["label"])
        sd = out / "samples" / sid
        p0, p90, p180 = sd / "emit_t0.tif", sd / "emit_tminus90.tif", sd / "emit_tminus180.tif"
        old = prev.get(sid)

        if old and old.get("status") == "PASS" and all(valid_tif(v4, p) for p in (p0, p90, p180)):
            print(f"[{i:03d}/{len(work):03d}] {sid} -> RESUME SKIP PASS")
            progress.append(old)
            built.append({
                "id": sid, "label": label, "latitude": r["latitude"], "longitude": r["longitude"],
                "emit_0_path": str(p0.relative_to(out)), "emit_90_path": str(p90.relative_to(out)),
                "emit_360_path": str(p180.relative_to(out)), "pair_id": pid,
                "scene_role": "positive" if label == 1 else "candidate_negative",
                "source_t0_utc": r["source_t0_utc"], "source_ch4enh_scene_id": r.get("source_ch4enh_scene_id", ""),
                "temporal_mode": "real_t0_tminus90_tminus180",
                "third_frame_semantics": "actual_tminus180_aliased_to_emit_360_path",
                "t90_actual_days_before_t0": old.get("t90_actual_days_before_t0", ""),
                "t90_target_error_days": old.get("t90_target_error_days", ""),
                "t180_actual_days_before_t0": old.get("t180_actual_days_before_t0", ""),
                "t180_target_error_days": old.get("t180_target_error_days", ""),
                "t90_l2a_granule": old.get("t90_l2a_granule", ""),
                "t180_l2a_granule": old.get("t180_l2a_granule", ""),
            })
            continue

        print(f"[{i:03d}/{len(work):03d}] {sid}")
        shutil.rmtree(sd, ignore_errors=True)
        sd.mkdir(parents=True, exist_ok=True)
        src_t0 = root / "samples" / sid / "emit_t0.tif"
        if not valid_tif(v4, src_t0):
            rec = {"id": sid, "pair_id": pid, "label": label, "status": "FAIL", "reason": "invalid_source_t0"}
            progress.append(rec); write_csv(progress_path, progress)
            print("  -> FAIL invalid t0")
            continue
        shutil.copy2(src_t0, p0)

        t0 = parse_dt(r["source_t0_utc"])
        lat, lon = float(r["latitude"]), float(r["longitude"])
        try:
            f90 = build_frame(v4, lon, lat, t0, 90, limit_days, set(), srf, raw, p90)
            f180 = build_frame(v4, lon, lat, t0, 180, limit_days, {f90["granule"]}, srf, raw, p180)
            rec = {
                "id": sid, "pair_id": pid, "label": label, "status": "PASS", "reason": "",
                "t90_l2a_granule": f90["granule"], "t90_utc": f90["utc"],
                "t90_actual_days_before_t0": f"{f90['actual']:.3f}", "t90_target_error_days": f"{f90['error']:.3f}",
                "t180_l2a_granule": f180["granule"], "t180_utc": f180["utc"],
                "t180_actual_days_before_t0": f"{f180['actual']:.3f}", "t180_target_error_days": f"{f180['error']:.3f}",
            }
            progress.append(rec)
            built.append({
                "id": sid, "label": label, "latitude": r["latitude"], "longitude": r["longitude"],
                "emit_0_path": str(p0.relative_to(out)), "emit_90_path": str(p90.relative_to(out)),
                "emit_360_path": str(p180.relative_to(out)), "pair_id": pid,
                "scene_role": "positive" if label == 1 else "candidate_negative",
                "source_t0_utc": r["source_t0_utc"], "source_ch4enh_scene_id": r.get("source_ch4enh_scene_id", ""),
                "temporal_mode": "real_t0_tminus90_tminus180",
                "third_frame_semantics": "actual_tminus180_aliased_to_emit_360_path",
                "t90_actual_days_before_t0": rec["t90_actual_days_before_t0"],
                "t90_target_error_days": rec["t90_target_error_days"],
                "t180_actual_days_before_t0": rec["t180_actual_days_before_t0"],
                "t180_target_error_days": rec["t180_target_error_days"],
                "t90_l2a_granule": rec["t90_l2a_granule"], "t180_l2a_granule": rec["t180_l2a_granule"],
            })
            print("  -> PASS")
        except Exception as exc:
            v4.cleanup_raw_l2a_cache(raw)
            shutil.rmtree(sd, ignore_errors=True)
            progress.append({"id": sid, "pair_id": pid, "label": label, "status": "FAIL", "reason": str(exc)})
            print(f"  -> FAIL: {exc}")

        write_csv(progress_path, progress)
        write_csv(out / "eval_emit_480m_temporal_all_pass.partial.csv", built)

    v4.cleanup_raw_l2a_cache(raw)
    pair_map = {}
    for r in built:
        pair_map.setdefault(r["pair_id"], []).append(r)
    final_pairs = {pid for pid, g in pair_map.items() if len(g) == 2 and {int(x["label"]) for x in g} == {0, 1}}
    final = [r for r in built if r["pair_id"] in final_pairs]

    for r in built:
        if r["pair_id"] not in final_pairs:
            shutil.rmtree(out / "samples" / r["id"], ignore_errors=True)

    write_csv(out / "eval_emit_480m_temporal_all_pass.csv", built)
    write_csv(out / "eval_emit_480m.csv", final)
    write_csv(progress_path, progress)

    (out / "README_FOR_SENIOR.md").write_text(
        f"""# EMIT MethaneFuse REAL temporal dataset\n\n"
        f"Tier: {a.tier}\nTarget tolerance: +/-{limit_days:.0f} days\n"
        f"Final complete pairs: {len(final_pairs)}\nFinal samples: {len(final)}\n"
        f"Positive: {sum(int(r['label']) == 1 for r in final)}\n"
        f"Negative: {sum(int(r['label']) == 0 for r in final)}\n\n"
        "Temporal inputs are REAL separate EMIT L2A acquisitions.\n"
        "emit_0_path = real t0\n"
        "emit_90_path = real acquisition near t0-90\n"
        "emit_360_path = real acquisition near t0-180 (compatibility column name)\n\n"
        "All model TIFFs are EMIT L2A reflectance -> 16 WV3-like bands -> 480 m -> 518x518.\n"
        "Candidate negatives are same-site different-time observations without a published CH4PLM detection, not independently confirmed zero-emission observations.\n"
        """,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("BUILD COMPLETE")
    print("=" * 72)
    print(f"Search-ready complete pairs: {len(complete)}")
    print(f"Final complete pairs after pixel QA: {len(final_pairs)}")
    print(f"Final rows: {len(final)}")
    print(f"Positive: {sum(int(r['label']) == 1 for r in final)}")
    print(f"Negative: {sum(int(r['label']) == 0 for r in final)}")
    print(f"Manifest: {out/'eval_emit_480m.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-v4", default="prepare_emit_for_methanefuse_v4_resume_rflonly.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("--input-root", default="EMIT_MethaneFuse_480m_SMOKE_HANDOFF")
    s.add_argument("--search-out", default="emit_temporal_search")

    b = sub.add_parser("build")
    b.add_argument("--input-root", default="EMIT_MethaneFuse_480m_SMOKE_HANDOFF")
    b.add_argument("--search-csv", default="emit_temporal_search/temporal_selection.csv")
    b.add_argument("--tier", choices=("strict30", "expanded60", "all120"), default="expanded60")
    b.add_argument("--out", default="EMIT_MethaneFuse_480m_TEMPORAL")

    a = ap.parse_args()
    if a.cmd == "search":
        cmd_search(a)
    else:
        cmd_build(a)


if __name__ == "__main__":
    main()
