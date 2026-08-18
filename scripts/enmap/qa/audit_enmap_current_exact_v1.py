#!/usr/bin/env python3
"""
audit_enmap_current_exact_v1.py

Audit CURRENT EnMAP L2A catalogue resolution for the primary <=72h nominal
manifest without downloading imagery.

Critical rule:
  A current replacement is accepted ONLY when both
    - enmap:datatakeID matches the original datatake, AND
    - enmap:tileID matches the original tile.

There is NO fallback to another acquisition in the same time window.

Default input:
  ~/methane_release_project/enmap_download_phases_v1/
      03_phase2_primary_total_AB_nominal.csv

Output:
  ~/methane_release_project/enmap_exact_resolution_audit_v1/
      exact_resolution_limit5.csv
      exact_resolution_summary.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEMS = f"{STAC_ROOT}collections/{COLLECTION}/items"
USER_AGENT = "UAlberta-EnMAP-Exact-Resolver-Audit/1.0"

DEFAULT_MANIFEST = (
    Path.home()
    / "methane_release_project"
    / "enmap_download_phases_v1"
    / "03_phase2_primary_total_AB_nominal.csv"
)
DEFAULT_OUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_exact_resolution_audit_v1"
)

SCENE_RE = re.compile(
    r"DT(?P<datatake>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


def get_json(url: str, retries: int = 6, timeout: int = 90) -> Dict[str, Any]:
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/geo+json, application/json",
                    "Connection": "close",
                },
            )
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as e:
            last = e
            if attempt == retries:
                break
            wait = min(20, 2 ** (attempt - 1))
            print(
                f"  STAC retry {attempt}/{retries}: "
                f"{type(e).__name__}: {e}; sleep {wait}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"STAC request failed: {url}\n{last}")


def parse_scene(scene_id: str) -> Dict[str, Optional[str]]:
    m = SCENE_RE.search(str(scene_id))
    if not m:
        return {
            "datatake": None,
            "acq": None,
            "tile": None,
            "version": None,
            "proc": None,
        }
    return m.groupdict()


def normalize_datatake(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    digits = re.sub(r"\D", "", str(value))
    return digits.lstrip("0") or "0" if digits else None


def normalize_tile(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return str(int(digits))


def feature_datatake(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = (
        p.get("enmap:datatakeID")
        or p.get("enmap:datatakeId")
        or p.get("enmap:datatake_id")
        or p.get("datatakeID")
    )
    if v is not None:
        return normalize_datatake(v)
    return normalize_datatake(parse_scene(str(feat.get("id") or ""))["datatake"])


def feature_tile(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = p.get("enmap:tileID")
    if v is not None:
        return normalize_tile(v)
    return normalize_tile(parse_scene(str(feat.get("id") or ""))["tile"])


def candidate_processing_key(feat: Dict[str, Any]):
    sid = str(feat.get("id") or "")
    parsed = parse_scene(sid)
    p = feat.get("properties", {}) or {}
    # Prefer latest product generation / processing timestamp.
    return (
        str(parsed.get("proc") or ""),
        str(p.get("updated") or ""),
        str(p.get("created") or ""),
        sid,
    )


def metadata_href(feat: Dict[str, Any]) -> Optional[str]:
    assets = feat.get("assets", {}) or {}
    for key in ("metadata", "METADATA"):
        if key in assets and assets[key].get("href"):
            return str(assets[key]["href"])
    for key, asset in assets.items():
        text = (
            str(key) + " "
            + str(asset.get("title") or "") + " "
            + str(asset.get("href") or "")
        ).upper()
        if "METADATA" in text and asset.get("href"):
            return str(asset["href"])
    return None


def query_time_window(acq_text: str) -> List[Dict[str, Any]]:
    t = pd.to_datetime(
        acq_text,
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    if pd.isna(t):
        return []

    t0 = t - pd.Timedelta(minutes=20)
    t1 = t + pd.Timedelta(minutes=20)
    params = urlencode(
        {
            "datetime": (
                f"{t0.isoformat().replace('+00:00', 'Z')}/"
                f"{t1.isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 100,
            "f": "json",
        }
    )
    doc = get_json(f"{STAC_ITEMS}?{params}")
    return doc.get("features", []) or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, low_memory=False)
    df = df.drop_duplicates("l2a_scene_id").copy()

    if args.limit > 0:
        df = df.head(args.limit).copy()

    rows = []

    print("=" * 88)
    print("ENMAP CURRENT EXACT-ACQUISITION RESOLUTION AUDIT")
    print("=" * 88)
    print(f"Input scenes: {len(df)}")
    print()

    for pos, (_, r) in enumerate(df.iterrows(), start=1):
        original = str(r["l2a_scene_id"]).strip()
        parsed = parse_scene(original)

        wanted_dt = normalize_datatake(parsed["datatake"])
        wanted_tile = normalize_tile(parsed["tile"])
        acq = parsed["acq"]

        print(f"[{pos}/{len(df)}] {original}")
        print(f"  original datatake : {wanted_dt}")
        print(f"  original tile     : {wanted_tile}")

        if not wanted_dt or not wanted_tile or not acq:
            print("  STATUS            : UNPARSEABLE_ORIGINAL_ID")
            rows.append(
                {
                    "original_scene_id": original,
                    "original_datatake": wanted_dt,
                    "original_tile": wanted_tile,
                    "status": "UNPARSEABLE_ORIGINAL_ID",
                    "exact_candidate_count": 0,
                }
            )
            continue

        feats = query_time_window(acq)

        exact = [
            feat
            for feat in feats
            if feature_datatake(feat) == wanted_dt
            and feature_tile(feat) == wanted_tile
        ]

        print(f"  time-window items : {len(feats)}")
        print(f"  exact DT+tile     : {len(exact)}")

        if not exact:
            print("  STATUS            : NO_CURRENT_EXACT_PRODUCT")
            rows.append(
                {
                    "original_scene_id": original,
                    "original_datatake": wanted_dt,
                    "original_tile": wanted_tile,
                    "status": "NO_CURRENT_EXACT_PRODUCT",
                    "time_window_candidate_count": len(feats),
                    "exact_candidate_count": 0,
                }
            )
            continue

        exact = sorted(
            exact,
            key=candidate_processing_key,
            reverse=True,
        )
        current = exact[0]
        current_id = str(current.get("id") or "")
        current_dt = feature_datatake(current)
        current_tile = feature_tile(current)
        href = metadata_href(current)

        print(f"  CURRENT PRODUCT   : {current_id}")
        print(f"  current datatake  : {current_dt}")
        print(f"  current tile      : {current_tile}")
        print(
            "  same acquisition  : "
            + str(current_dt == wanted_dt and current_tile == wanted_tile)
        )
        print(f"  metadata href     : {href}")
        print("  STATUS            : EXACT_CURRENT_PRODUCT_FOUND")

        rows.append(
            {
                "original_scene_id": original,
                "original_datatake": wanted_dt,
                "original_tile": wanted_tile,
                "time_window_candidate_count": len(feats),
                "exact_candidate_count": len(exact),
                "current_scene_id": current_id,
                "current_datatake": current_dt,
                "current_tile": current_tile,
                "same_acquisition": (
                    current_dt == wanted_dt
                    and current_tile == wanted_tile
                ),
                "metadata_href": href,
                "status": "EXACT_CURRENT_PRODUCT_FOUND",
            }
        )
        print()

    out = pd.DataFrame(rows)
    csv_path = out_dir / (
        f"exact_resolution_limit{len(df)}.csv"
    )
    out.to_csv(csv_path, index=False)

    status_counts = out["status"].value_counts(dropna=False)

    summary_lines = [
        "ENMAP CURRENT EXACT-RESOLUTION SUMMARY",
        "=" * 88,
        f"Input scenes                  : {len(out)}",
        f"Exact current products found  : "
        f"{int((out['status'] == 'EXACT_CURRENT_PRODUCT_FOUND').sum())}",
        f"No current exact product      : "
        f"{int((out['status'] == 'NO_CURRENT_EXACT_PRODUCT').sum())}",
        "",
        "STATUS COUNTS",
    ]
    for k, v in status_counts.items():
        summary_lines.append(f"{str(k):36s} {int(v)}")

    summary_lines += [
        "",
        "IMPORTANT",
        "- A replacement is accepted only if datatake AND tile both match.",
        "- There is no fallback to another EnMAP acquisition in the same time window.",
        f"- Output: {csv_path}",
    ]

    summary_path = out_dir / "exact_resolution_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
