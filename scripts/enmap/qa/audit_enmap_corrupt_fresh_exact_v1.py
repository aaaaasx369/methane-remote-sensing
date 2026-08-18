#!/usr/bin/env python3
"""
audit_enmap_corrupt_fresh_exact_v1.py

For EnMAP assets currently marked CORRUPT_PART_PREFIX, refresh the EnMAP STAC
catalogue by exact acquisition identity (datatake + tile), WITHOUT using the
old local STAC cache.

Purpose:
- Determine whether the checkpoint's current_scene_id is still the current
  product.
- If the fresh current_scene_id changed, DO NOT splice bytes from the fresh
  product into the old partial; the binary may differ after reprocessing.
- If the scene ID is unchanged, the old URLs are stale and a fresh href can
  be tested safely in a later byte-comparison step.

Default input:
  ~/methane_release_project/enmap_primary721_download_v2/
      download_checkpoint_full.csv

Default output:
  ~/methane_release_project/enmap_corrupt_fresh_exact_audit_v1/
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
USER_AGENT = "UAlberta-EnMAP-Corrupt-Fresh-Exact-Audit/1.0"

DEFAULT_CHECKPOINT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary721_download_v2"
    / "download_checkpoint_full.csv"
)

DEFAULT_OUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_corrupt_fresh_exact_audit_v1"
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


def normalize_digits(value: Any) -> Optional[str]:
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
    return digits.lstrip("0") or "0"


def normalize_tile(value: Any) -> Optional[str]:
    d = normalize_digits(value)
    return str(int(d)) if d is not None else None


def feature_datatake(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = (
        p.get("enmap:datatakeID")
        or p.get("enmap:datatakeId")
        or p.get("enmap:datatake_id")
        or p.get("datatakeID")
    )
    if v is not None:
        return normalize_digits(v)
    return normalize_digits(
        parse_scene(str(feat.get("id") or "")).get("datatake")
    )


def feature_tile(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = p.get("enmap:tileID")
    if v is not None:
        return normalize_tile(v)
    return normalize_tile(
        parse_scene(str(feat.get("id") or "")).get("tile")
    )


def candidate_processing_key(feat: Dict[str, Any]):
    sid = str(feat.get("id") or "")
    parsed = parse_scene(sid)
    p = feat.get("properties", {}) or {}
    return (
        str(parsed.get("proc") or ""),
        str(p.get("updated") or ""),
        str(p.get("created") or ""),
        sid,
    )


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


def asset_href(feat: Dict[str, Any], asset_key: str) -> Optional[str]:
    assets = feat.get("assets", {}) or {}

    if asset_key in assets and assets[asset_key].get("href"):
        return str(assets[asset_key]["href"])

    # Fallback for image asset naming differences.
    wanted = str(asset_key).lower()
    for key, asset in assets.items():
        href = str(asset.get("href") or "")
        text = " ".join([
            str(key),
            str(asset.get("title") or ""),
            href,
        ]).lower()

        if wanted == "image" and "spectral_image" in text and href:
            return href

        if wanted in text and href:
            return href

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    df = pd.read_csv(checkpoint, low_memory=False)

    bad = df[
        df["download_status"]
        .fillna("")
        .astype(str)
        .str.startswith("CORRUPT_PART_PREFIX:")
    ].copy()

    # One row per corrupt asset. Keep asset-level distinction because a scene
    # can have more than one TIFF quality/mask asset.
    if args.limit > 0:
        bad = bad.head(args.limit).copy()

    rows = []

    print("=" * 96)
    print("ENMAP CORRUPT PARTIALS — FRESH EXACT-PRODUCT AUDIT")
    print("=" * 96)
    print("Rows:", len(bad))
    print()

    for pos, (_, r) in enumerate(bad.iterrows(), start=1):
        original = str(r["original_scene_id"]).strip()
        old_current = str(r["current_scene_id"]).strip()
        asset_key = str(r["asset_key"]).strip()

        parsed = parse_scene(original)
        wanted_dt = normalize_digits(parsed.get("datatake"))
        wanted_tile = normalize_tile(parsed.get("tile"))
        acq = parsed.get("acq")

        print(f"[{pos}/{len(bad)}] {old_current} :: {asset_key}")
        print(f"  original DT/tile : {wanted_dt}/{wanted_tile}")

        base = {
            "original_scene_id": original,
            "old_current_scene_id": old_current,
            "asset_key": asset_key,
            "wanted_datatake": wanted_dt,
            "wanted_tile": wanted_tile,
        }

        if not wanted_dt or not wanted_tile or not acq:
            print("  STATUS           : UNPARSEABLE_ORIGINAL_ID")
            rows.append({
                **base,
                "status": "UNPARSEABLE_ORIGINAL_ID",
            })
            continue

        try:
            feats = query_time_window(acq)
        except Exception as e:
            print(f"  STATUS           : STAC_ERROR {type(e).__name__}: {e}")
            rows.append({
                **base,
                "status": "STAC_ERROR",
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        exact = [
            feat
            for feat in feats
            if feature_datatake(feat) == wanted_dt
            and feature_tile(feat) == wanted_tile
        ]

        if not exact:
            print("  STATUS           : NO_CURRENT_EXACT_PRODUCT")
            rows.append({
                **base,
                "status": "NO_CURRENT_EXACT_PRODUCT",
                "time_window_items": len(feats),
                "exact_candidate_count": 0,
            })
            continue

        exact = sorted(
            exact,
            key=candidate_processing_key,
            reverse=True,
        )

        fresh = exact[0]
        fresh_id = str(fresh.get("id") or "")
        fresh_href = asset_href(fresh, asset_key)
        same_id = fresh_id == old_current

        print(f"  fresh current ID : {fresh_id}")
        print(f"  same as old      : {same_id}")
        print(f"  fresh asset href : {fresh_href}")
        print(
            "  STATUS           : "
            + (
                "SAME_PRODUCT_FRESH_HREF"
                if same_id
                else "PRODUCT_REPROCESSED_OR_REPLACED"
            )
        )
        print()

        rows.append({
            **base,
            "fresh_current_scene_id": fresh_id,
            "same_current_scene_id": same_id,
            "fresh_asset_href": fresh_href,
            "time_window_items": len(feats),
            "exact_candidate_count": len(exact),
            "status": (
                "SAME_PRODUCT_FRESH_HREF"
                if same_id
                else "PRODUCT_REPROCESSED_OR_REPLACED"
            ),
        })

    out = pd.DataFrame(rows)

    csv_path = out_dir / (
        f"fresh_exact_audit_limit{len(out)}.csv"
    )
    out.to_csv(csv_path, index=False)

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    if len(out):
        print(out["status"].value_counts(dropna=False).to_string())
    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
