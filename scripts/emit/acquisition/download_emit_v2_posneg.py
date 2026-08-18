#!/usr/bin/env python3
"""
Build a balanced EMIT V2 methane dataset:
- 50 positive EMITL2BCH4ENH.002 scenes referenced by EMITL2BCH4PLM.002
- 50 paired candidate-negative EMITL2BCH4ENH.002 scenes
  at the same geographic point, close in time, and not referenced by any
  currently published EMITL2BCH4PLM.002 plume complex.

IMPORTANT:
A "candidate negative" here means "no published high-confidence CH4PLM plume
complex references this scene." It is NOT guaranteed to contain zero methane.
Keep the candidate-negative flag until a later QA step.

Usage:
    python download_emit_v2_posneg.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import earthaccess
except ImportError:
    print("ERROR: earthaccess is not installed.")
    print("Run: python3 -m pip install -U earthaccess")
    sys.exit(1)

# -----------------------------
# Configuration
# -----------------------------
N_PAIRS = 50
RANDOM_SEED = 42
NEGATIVE_SEARCH_WINDOWS_DAYS = (30, 90, 180, 365)
MAX_NEGATIVE_RESULTS_PER_SEARCH = 300

ROOT = Path("emit_v2_posneg_100")
PLM_META_ALL = ROOT / "00_all_plm_json"
POS_ENH_DIR = ROOT / "01_positive_ch4enh"
NEG_ENH_DIR = ROOT / "02_candidate_negative_ch4enh"
POS_PLM_DIR = ROOT / "03_positive_plm_labels"

PAIRS_CSV = ROOT / "emit_v2_pairs.csv"
INVENTORY_CSV = ROOT / "emit_v2_inventory.csv"
README_TXT = ROOT / "README.txt"

for d in (ROOT, PLM_META_ALL, POS_ENH_DIR, NEG_ENH_DIR, POS_PLM_DIR):
    d.mkdir(parents=True, exist_ok=True)

FULL_SCENE_RE = re.compile(
    r"EMIT_L2B_CH4ENH_002_(\d{8}T\d{6})_(\d+)_(\d{3})"
)
CORE_SCENE_RE = re.compile(r"(\d{8}T\d{6})_(\d+)_(\d{3})")
PLM_RE = re.compile(r"EMIT_L2B_CH4PLM_002_(\d{8}T\d{6})_(\d{6})")


def recursive_strings(obj: Any) -> Iterable[str]:
    """Yield every string nested inside dict/list structures."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from recursive_strings(k)
            yield from recursive_strings(v)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from recursive_strings(x)


def extract_scene_ids(obj: Any) -> set[str]:
    """
    Extract V2 CH4ENH scene identifiers from arbitrary metadata.

    First look for full EMIT_L2B_CH4ENH_002_* names.
    Fallback: find timestamp_orbit_scene patterns and construct CH4ENH names.
    """
    out: set[str] = set()
    all_strings = list(recursive_strings(obj))

    for s in all_strings:
        for m in FULL_SCENE_RE.finditer(s):
            out.add(f"EMIT_L2B_CH4ENH_002_{m.group(1)}_{m.group(2)}_{m.group(3)}")

    if out:
        return out

    # Fallback for metadata that stores source scenes without the CH4ENH prefix.
    for s in all_strings:
        for m in CORE_SCENE_RE.finditer(s):
            # Avoid accidentally converting CH4PLM names:
            # plume-complex identifiers are six digits rather than a 3-digit scene.
            out.add(f"EMIT_L2B_CH4ENH_002_{m.group(1)}_{m.group(2)}_{m.group(3)}")

    return out


def extract_plm_name(obj: Any) -> str | None:
    for s in recursive_strings(obj):
        m = PLM_RE.search(s)
        if m:
            return f"EMIT_L2B_CH4PLM_002_{m.group(1)}_{m.group(2)}"
    return None


def scene_datetime(scene_id: str) -> datetime:
    m = FULL_SCENE_RE.search(scene_id)
    if not m:
        raise ValueError(f"Cannot parse scene datetime: {scene_id}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def result_scene_id(granule: Any) -> str | None:
    """Extract CH4ENH scene ID from an earthaccess DataGranule."""
    try:
        found = extract_scene_ids(granule)
        if found:
            # A CH4ENH granule should contain one scene identity.
            return sorted(found)[0]
    except Exception:
        pass

    try:
        for url in granule.data_links():
            found = extract_scene_ids(url)
            if found:
                return sorted(found)[0]
    except Exception:
        pass

    return None


def result_plm_name(granule: Any) -> str | None:
    try:
        name = extract_plm_name(granule)
        if name:
            return name
    except Exception:
        pass

    try:
        for url in granule.data_links():
            name = extract_plm_name(url)
            if name:
                return name
    except Exception:
        pass

    return None


def granule_center(granule: Any) -> tuple[float, float]:
    """
    Return an approximate (lon, lat) center from UMM-G spatial metadata.
    Handles polygon or bounding-rectangle geometry.
    """
    umm = granule.get("umm", granule)
    geom = (
        umm.get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
    )

    polygons = geom.get("GPolygons") or []
    if polygons:
        pts = polygons[0].get("Boundary", {}).get("Points", [])
        if pts:
            lons = [float(p["Longitude"]) for p in pts]
            lats = [float(p["Latitude"]) for p in pts]
            return sum(lons) / len(lons), sum(lats) / len(lats)

    rects = geom.get("BoundingRectangles") or []
    if rects:
        r = rects[0]
        west = float(r["WestBoundingCoordinate"])
        east = float(r["EastBoundingCoordinate"])
        south = float(r["SouthBoundingCoordinate"])
        north = float(r["NorthBoundingCoordinate"])
        return (west + east) / 2.0, (south + north) / 2.0

    raise ValueError("No usable UMM spatial polygon/bounding rectangle found.")


def plm_json_url(granule: Any) -> str | None:
    """Return the PLM GeoJSON/JSON URL, deriving it from the TIF link if needed."""
    try:
        links = list(granule.data_links())
    except Exception:
        links = []

    https_links = [u for u in links if isinstance(u, str) and u.startswith("http")]

    for u in https_links:
        if re.search(r"\.json(?:\?.*)?$", u, flags=re.I):
            return u

    for u in https_links:
        if re.search(r"\.tif(?:\?.*)?$", u, flags=re.I):
            return re.sub(r"\.tif(?=(?:\?.*)?$)", ".json", u, flags=re.I)

    return None


def search_enh_scene(scene_id: str):
    """Find one exact EMITL2BCH4ENH.002 granule."""
    attempts = [
        scene_id,
        scene_id + "*",
    ]
    for name in attempts:
        hits = earthaccess.search_data(
            short_name="EMITL2BCH4ENH",
            version="002",
            granule_name=name,
            count=10,
        )
        for g in hits:
            sid = result_scene_id(g)
            if sid == scene_id:
                return g
        if len(hits) == 1:
            return hits[0]
    return None


def find_negative_for_positive(
    pos_scene_id: str,
    pos_granule: Any,
    all_plume_scene_ids: set[str],
    used_negative_ids: set[str],
):
    """
    Search same geographic point, close in time.
    Exclude every scene referenced by any published CH4PLM V2 complex.
    """
    lon, lat = granule_center(pos_granule)
    pos_dt = scene_datetime(pos_scene_id)

    for window_days in NEGATIVE_SEARCH_WINDOWS_DAYS:
        start = (pos_dt - timedelta(days=window_days)).isoformat()
        end = (pos_dt + timedelta(days=window_days)).isoformat()

        hits = earthaccess.search_data(
            short_name="EMITL2BCH4ENH",
            version="002",
            point=(lon, lat),
            temporal=(start, end),
            count=MAX_NEGATIVE_RESULTS_PER_SEARCH,
        )

        candidates = []
        for g in hits:
            sid = result_scene_id(g)
            if not sid:
                continue
            if sid == pos_scene_id:
                continue
            if sid in all_plume_scene_ids:
                continue
            if sid in used_negative_ids:
                continue

            try:
                dt = scene_datetime(sid)
            except Exception:
                continue

            delta_days = abs((dt - pos_dt).total_seconds()) / 86400.0

            # Avoid using essentially the same acquisition a few minutes later.
            if delta_days < 0.5:
                continue

            candidates.append((delta_days, sid, g, dt))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            delta_days, sid, g, dt = candidates[0]
            return {
                "scene_id": sid,
                "granule": g,
                "datetime": dt,
                "delta_days": delta_days,
                "lon": lon,
                "lat": lat,
                "search_window_days": window_days,
            }

    return None


def write_csvs(pairs: list[dict[str, Any]]) -> None:
    pair_fields = [
        "pair_id",
        "positive_scene_id",
        "negative_scene_id",
        "positive_time_utc",
        "negative_time_utc",
        "delta_days",
        "match_lon",
        "match_lat",
        "negative_search_window_days",
        "positive_plm_granules",
        "positive_label",
        "negative_label",
        "negative_label_type",
        "matching_rule",
    ]

    with PAIRS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        for p in pairs:
            w.writerow({
                "pair_id": p["pair_id"],
                "positive_scene_id": p["positive_scene_id"],
                "negative_scene_id": p["negative_scene_id"],
                "positive_time_utc": p["positive_time"].isoformat(),
                "negative_time_utc": p["negative_time"].isoformat(),
                "delta_days": f'{p["delta_days"]:.3f}',
                "match_lon": f'{p["lon"]:.6f}',
                "match_lat": f'{p["lat"]:.6f}',
                "negative_search_window_days": p["search_window_days"],
                "positive_plm_granules": ";".join(p["plm_names"]),
                "positive_label": 1,
                "negative_label": 0,
                "negative_label_type": "candidate_negative_no_published_CH4PLM_reference",
                "matching_rule": "same_point_near_time_exclude_all_published_CH4PLM_source_scenes",
            })

    inv_fields = [
        "sample_id",
        "pair_id",
        "label",
        "scene_id",
        "acquisition_time_utc",
        "match_lon",
        "match_lat",
        "label_source",
        "label_strength",
    ]
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields)
        w.writeheader()
        for p in pairs:
            w.writerow({
                "sample_id": f'{p["pair_id"]}_POS',
                "pair_id": p["pair_id"],
                "label": 1,
                "scene_id": p["positive_scene_id"],
                "acquisition_time_utc": p["positive_time"].isoformat(),
                "match_lon": f'{p["lon"]:.6f}',
                "match_lat": f'{p["lat"]:.6f}',
                "label_source": "EMITL2BCH4PLM.002",
                "label_strength": "high_confidence_published_plume",
            })
            w.writerow({
                "sample_id": f'{p["pair_id"]}_NEG',
                "pair_id": p["pair_id"],
                "label": 0,
                "scene_id": p["negative_scene_id"],
                "acquisition_time_utc": p["negative_time"].isoformat(),
                "match_lon": f'{p["lon"]:.6f}',
                "match_lat": f'{p["lat"]:.6f}',
                "label_source": "absence_from_all_published_EMITL2BCH4PLM.002_source_scenes",
                "label_strength": "candidate_negative_requires_QA",
            })


def main():
    print("=" * 72)
    print("EMIT V2 positive / candidate-negative downloader")
    print("=" * 72)

    # 1) Login
    print("\n[1/7] Earthdata login")
    try:
        earthaccess.login(strategy="netrc")
        print("Authenticated using ~/.netrc")
    except Exception:
        print("No usable ~/.netrc login found; switching to interactive login.")
        earthaccess.login(strategy="interactive", persist=False)
        print("Authenticated for this run (credentials not persisted by this script).")

    # 2) Search ALL CH4PLM V2 granules
    print("\n[2/7] Searching all EMITL2BCH4PLM.002 granules...")
    plm_results = earthaccess.search_data(
        short_name="EMITL2BCH4PLM",
        version="002",
        count=-1,
    )
    print(f"CH4PLM V2 granules found: {len(plm_results)}")
    if not plm_results:
        raise RuntimeError("No EMITL2BCH4PLM.002 granules found.")

    # Map PLM granule names to search results.
    plm_result_by_name = {}
    for g in plm_results:
        name = result_plm_name(g)
        if name:
            plm_result_by_name[name] = g

    # 3) Download ALL small PLM JSON metadata only.
    print("\n[3/7] Collecting PLM JSON metadata URLs...")
    json_urls = []
    for g in plm_results:
        u = plm_json_url(g)
        if u:
            json_urls.append(u)
    json_urls = sorted(set(json_urls))
    print(f"PLM JSON URLs found/derived: {len(json_urls)}")

    if not json_urls:
        raise RuntimeError("Could not find/derive PLM JSON URLs.")

    print("Downloading PLM JSON metadata (small files; needed to build the global plume-scene exclusion set)...")
    earthaccess.download(
        json_urls,
        local_path=str(PLM_META_ALL),
        provider="LPCLOUD",
        threads=8,
    )

    # 4) Parse global set of scenes that have any published plume complex.
    print("\n[4/7] Parsing plume metadata...")
    all_plume_scene_ids: set[str] = set()
    scene_to_plm_names: dict[str, set[str]] = defaultdict(set)
    parsed_json = 0

    for fp in sorted(PLM_META_ALL.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: could not parse {fp.name}: {e}")
            continue

        parsed_json += 1
        scenes = extract_scene_ids(obj)
        plm_name = extract_plm_name(fp.name) or fp.stem

        for sid in scenes:
            all_plume_scene_ids.add(sid)
            scene_to_plm_names[sid].add(plm_name)

    print(f"PLM JSON parsed: {parsed_json}")
    print(f"Unique CH4ENH source scenes with published plume complexes: {len(all_plume_scene_ids)}")

    if len(all_plume_scene_ids) < N_PAIRS:
        raise RuntimeError(
            f"Only {len(all_plume_scene_ids)} plume source scenes found; "
            f"need at least {N_PAIRS}. Inspect the downloaded JSON schema."
        )

    # 5) Build 50 same-point / near-time positive-negative pairs.
    print(f"\n[5/7] Building {N_PAIRS} matched positive-negative pairs...")
    positive_pool = sorted(all_plume_scene_ids)
    random.Random(RANDOM_SEED).shuffle(positive_pool)

    pairs: list[dict[str, Any]] = []
    pos_granules = []
    neg_granules = []
    used_positive_ids: set[str] = set()
    used_negative_ids: set[str] = set()

    for idx, pos_sid in enumerate(positive_pool, start=1):
        if len(pairs) >= N_PAIRS:
            break

        print(f"[candidate {idx}/{len(positive_pool)}] positive: {pos_sid}")

        pos_g = search_enh_scene(pos_sid)
        if pos_g is None:
            print("  -> skip: CH4ENH V2 granule not found")
            continue

        try:
            neg = find_negative_for_positive(
                pos_sid,
                pos_g,
                all_plume_scene_ids,
                used_negative_ids,
            )
        except Exception as e:
            print(f"  -> skip: negative search failed: {e}")
            continue

        if neg is None:
            print("  -> skip: no same-point candidate negative within 365 days")
            continue

        pair_id = f"EMITPAIR_{len(pairs)+1:03d}"
        plm_names = sorted(scene_to_plm_names.get(pos_sid, set()))

        pair = {
            "pair_id": pair_id,
            "positive_scene_id": pos_sid,
            "negative_scene_id": neg["scene_id"],
            "positive_time": scene_datetime(pos_sid),
            "negative_time": neg["datetime"],
            "delta_days": neg["delta_days"],
            "lon": neg["lon"],
            "lat": neg["lat"],
            "search_window_days": neg["search_window_days"],
            "plm_names": plm_names,
        }
        pairs.append(pair)
        pos_granules.append(pos_g)
        neg_granules.append(neg["granule"])
        used_positive_ids.add(pos_sid)
        used_negative_ids.add(neg["scene_id"])

        print(
            f"  -> ACCEPT {pair_id}: negative={neg['scene_id']} "
            f"(Δ={neg['delta_days']:.1f} days, window=±{neg['search_window_days']} d)"
        )

    if len(pairs) < N_PAIRS:
        print(
            f"\nWARNING: built only {len(pairs)} pairs, fewer than requested {N_PAIRS}."
        )
        print("The script will still download the pairs it found.")

    if not pairs:
        raise RuntimeError("No positive-negative pairs could be built.")

    write_csvs(pairs)
    print(f"Pair table: {PAIRS_CSV}")
    print(f"Inventory:  {INVENTORY_CSV}")

    # 6) Download CH4ENH for positive + candidate-negative scenes.
    print("\n[6/7] Downloading selected CH4ENH V2 granules...")
    print(f"Positive scenes: {len(pos_granules)} -> {POS_ENH_DIR}")
    earthaccess.download(
        pos_granules,
        local_path=str(POS_ENH_DIR),
        threads=8,
    )

    print(f"Candidate-negative scenes: {len(neg_granules)} -> {NEG_ENH_DIR}")
    earthaccess.download(
        neg_granules,
        local_path=str(NEG_ENH_DIR),
        threads=8,
    )

    # 7) Download PLM label granules that support the selected positives.
    print("\n[7/7] Downloading CH4PLM label granules for selected positives...")
    selected_plm_names = sorted({
        name
        for p in pairs
        for name in p["plm_names"]
        if name
    })

    selected_plm_results = []
    seen_plm = set()

    for name in selected_plm_names:
        g = plm_result_by_name.get(name)
        if g is None:
            hits = earthaccess.search_data(
                short_name="EMITL2BCH4PLM",
                version="002",
                granule_name=name,
                count=10,
            )
            g = hits[0] if hits else None

        if g is not None and name not in seen_plm:
            selected_plm_results.append(g)
            seen_plm.add(name)

    print(f"Selected PLM granules: {len(selected_plm_results)} -> {POS_PLM_DIR}")
    if selected_plm_results:
        earthaccess.download(
            selected_plm_results,
            local_path=str(POS_PLM_DIR),
            threads=8,
        )

    README_TXT.write_text(
        f"""EMIT V2 balanced methane dataset
================================

Requested pairs: {N_PAIRS}
Pairs built: {len(pairs)}

01_positive_ch4enh/
    EMITL2BCH4ENH.002 scenes that are referenced by one or more published
    EMITL2BCH4PLM.002 high-confidence plume complexes.

02_candidate_negative_ch4enh/
    Matched scenes intersecting the same geographic point, close in time,
    excluding every CH4ENH source scene referenced by the currently published
    CH4PLM V2 metadata set used during this run.

03_positive_plm_labels/
    Published CH4PLM V2 granules supporting the positive scenes.

00_all_plm_json/
    All downloaded CH4PLM V2 JSON metadata used to build the global exclusion set.

IMPORTANT LABEL NOTE
--------------------
Negative label 0 is a CANDIDATE / WEAK NEGATIVE, not a guaranteed physical
absence of methane. EMIT CH4PLM is a conservative high-confidence plume product,
so an unreported plume may still exist. Run image-level QA before treating these
as strict negatives.

Tables:
- emit_v2_pairs.csv
- emit_v2_inventory.csv
""",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Pairs built: {len(pairs)}")
    print(f"Output folder: {ROOT.resolve()}")
    print(f"Pair table:    {PAIRS_CSV.resolve()}")
    print(f"Inventory:     {INVENTORY_CSV.resolve()}")
    print("\nNext step: run QA on candidate-negative CH4ENH rasters before calling them strict negatives.")


if __name__ == "__main__":
    main()
