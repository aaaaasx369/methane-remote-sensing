#!/usr/bin/env python3
"""
download_enmap_primary72h_nominal_v3_fresh.py

Robust EnMAP PRIMARY downloader for:
  <=72 h + NOMINAL L2A

Why V3 exists
-------------
The EnMAP L2A ARD archive is being reprocessed. Older product versions can be
replaced, while a previously cached STAC item / asset URL may remain stale.
That can produce a 404 even though the acquisition still exists.

V3 therefore:
1. Resolves each manifest row AGAINST THE CURRENT L2A CATALOGUE using
   acquisition time + datatake + tile.
2. Prefers the newest matching current product.
3. Stores original_scene_id AND resolved_scene_id.
4. If a download returns HTTP 404, refreshes that scene live and retries the
   same asset once with the newly resolved URL.
5. Deletes tiny HTTP-error .part bodies before retrying, but preserves real
   partial downloads for resume.
6. Downloads directly to the lab SMB share.

No imagery is downloaded during --preflight-only.

Default manifest:
  ~/methane_release_project/enmap_download_phases_v1/
      03_phase2_primary_total_AB_nominal.csv

Default destination:
  /Volumes/engg-leung/dora lin/EnMAP_MethaneFuse/
      01_raw_L2A/primary72h_nominal/
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEMS = f"{STAC_ROOT}collections/{COLLECTION}/items"
STAC_ITEM_BASE = f"{STAC_ITEMS}/"

DEFAULT_PROJECT = Path.home() / "methane_release_project"
DEFAULT_MANIFEST = (
    DEFAULT_PROJECT
    / "enmap_download_phases_v1"
    / "03_phase2_primary_total_AB_nominal.csv"
)
DEFAULT_DEST = Path(
    "/Volumes/engg-leung/dora lin/EnMAP_MethaneFuse/"
    "01_raw_L2A/primary72h_nominal"
)
DEFAULT_STATE = DEFAULT_PROJECT / "enmap_primary72h_download_v3"
SMB_MOUNT = Path("/Volumes/engg-leung")

EXCLUDE_ROLES = {"overview", "thumbnail"}
USER_AGENT = "UAlberta-Methane-EnMAP-Downloader/3.0"

# Example:
# ENMAP01-____L2A-DT0000125491_20250422T105437Z_001_V010502_20250423T032300Z
SCENE_RE = re.compile(
    r"DT(?P<dt>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


# =============================================================================
# SYSTEM / NETWORK
# =============================================================================

def smb_is_mounted() -> bool:
    try:
        r = subprocess.run(
            ["mount"], capture_output=True, text=True, check=False
        )
        return f" on {SMB_MOUNT} " in r.stdout
    except Exception:
        return False


def require_smb():
    if not smb_is_mounted():
        print()
        print("=" * 80)
        print("SMB SHARE IS NOT MOUNTED")
        print("=" * 80)
        print(f"Expected mount: {SMB_MOUNT}")
        print("Reconnect the lab share, then rerun the SAME command.")
        print("Completed files and real partial downloads are preserved.")
        raise SystemExit(20)


def get_json(url: str, retries: int = 8, timeout: int = 120) -> Dict[str, Any]:
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
            wait = min(30, 2 ** (attempt - 1))
            print(
                f"  STAC retry {attempt}/{retries}: "
                f"{type(e).__name__}: {e}; sleep {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(f"STAC request failed:\n{url}\n{last}")


# =============================================================================
# ENMAP PRODUCT ID HELPERS
# =============================================================================

def parse_scene_id(scene_id: str) -> Dict[str, Optional[str]]:
    m = SCENE_RE.search(scene_id)
    if not m:
        return {
            "datatake": None,
            "acq": None,
            "tile": None,
            "version": None,
            "proc": None,
        }
    return {
        "datatake": m.group("dt"),
        "acq": m.group("acq"),
        "tile": m.group("tile"),
        "version": m.group("version"),
        "proc": m.group("proc"),
    }


def normalize_datatake(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.upper().replace("DT", "")
    digits = re.sub(r"\D", "", s)
    return digits.lstrip("0") or "0"


def feature_datatake(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    value = (
        p.get("enmap:datatakeID")
        or p.get("enmap:datatakeId")
        or p.get("enmap:datatake_id")
        or p.get("datatakeID")
    )
    if value is not None:
        return normalize_datatake(value)

    sid = str(feat.get("id") or "")
    return normalize_datatake(parse_scene_id(sid).get("datatake"))


def feature_tile(feat: Dict[str, Any]) -> Optional[str]:
    return parse_scene_id(str(feat.get("id") or "")).get("tile")


def feature_proc_sort_key(feat: Dict[str, Any]) -> Tuple[str, str]:
    sid = str(feat.get("id") or "")
    parsed = parse_scene_id(sid)
    p = feat.get("properties", {}) or {}
    created = str(p.get("created") or p.get("updated") or "")
    return (str(parsed.get("proc") or ""), created)


def targeted_candidates(
    original_scene_id: str,
    l2a_time: Any,
    l2a_datatake_id: Any,
) -> List[Dict[str, Any]]:
    """
    Query the CURRENT collection around the acquisition time and then match
    datatake + tile locally.
    """
    parsed = parse_scene_id(original_scene_id)

    t = pd.to_datetime(l2a_time, errors="coerce", utc=True)
    if pd.isna(t) and parsed.get("acq"):
        t = pd.to_datetime(
            parsed["acq"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
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
    feats = doc.get("features", []) or []

    wanted_dt = normalize_datatake(l2a_datatake_id)
    if wanted_dt is None:
        wanted_dt = normalize_datatake(parsed.get("datatake"))
    wanted_tile = parsed.get("tile")

    # Strong match: same datatake and same tile.
    strong = []
    for feat in feats:
        dt_ok = wanted_dt is None or feature_datatake(feat) == wanted_dt
        tile_ok = wanted_tile is None or feature_tile(feat) == wanted_tile
        if dt_ok and tile_ok:
            strong.append(feat)

    if strong:
        return strong

    # Fallback: same datatake.
    medium = []
    for feat in feats:
        if wanted_dt is None or feature_datatake(feat) == wanted_dt:
            medium.append(feat)
    if medium:
        return medium

    return feats


def direct_item(scene_id: str) -> Optional[Dict[str, Any]]:
    url = STAC_ITEM_BASE + quote(scene_id, safe="") + "?f=json"
    try:
        return get_json(url)
    except Exception:
        return None


def resolve_current_scene(
    row: pd.Series,
    cache_dir: Path,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Resolve the current product representing the original acquisition.
    """
    original_scene_id = str(row["l2a_scene_id"]).strip()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{original_scene_id}.json"

    if (
        not force_refresh
        and cache_path.exists()
        and cache_path.stat().st_size > 0
    ):
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    candidates = targeted_candidates(
        original_scene_id=original_scene_id,
        l2a_time=row.get("l2a_time"),
        l2a_datatake_id=row.get("l2a_datatake_id"),
    )

    if candidates:
        # During reprocessing multiple generations may coexist in catalogue
        # results. Prefer the newest processing timestamp / created time.
        candidates = sorted(
            candidates,
            key=feature_proc_sort_key,
            reverse=True,
        )
        doc = candidates[0]
    else:
        doc = direct_item(original_scene_id)
        if doc is None:
            raise RuntimeError(
                f"Could not resolve current EnMAP product for "
                f"{original_scene_id}"
            )

    tmp = cache_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)
    return doc


# =============================================================================
# ASSET PLAN
# =============================================================================

def asset_is_science_relevant(asset: Dict[str, Any]) -> bool:
    roles = set(asset.get("roles") or [])
    if roles & EXCLUDE_ROLES:
        return False

    href = str(asset.get("href") or "")
    if not href:
        return False

    if roles & {"data", "metadata", "quality", "data-mask"}:
        return True

    text = " ".join(
        [
            str(asset.get("title") or ""),
            str(asset.get("description") or ""),
            href,
        ]
    ).upper()

    keep_tokens = [
        "SPECTRAL_IMAGE",
        "METADATA",
        "QUALITY",
        "PIXELMASK",
        "DEFECTIVE",
    ]
    return any(tok in text for tok in keep_tokens)


def safe_filename_from_href(href: str) -> str:
    return href.split("?")[0].rstrip("/").split("/")[-1]


def extract_plan_rows(
    manifest: pd.DataFrame,
    cache_dir: Path,
    dest_root: Path,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for pos, (_, r) in enumerate(manifest.iterrows(), start=1):
        original_scene_id = str(r["l2a_scene_id"]).strip()
        print(
            f"[resolve {pos}/{len(manifest)}] "
            f"{original_scene_id}"
        )

        doc = resolve_current_scene(
            r, cache_dir, force_refresh=force_refresh
        )
        resolved_scene_id = str(doc.get("id") or original_scene_id)

        if resolved_scene_id != original_scene_id:
            print(f"  CURRENT PRODUCT -> {resolved_scene_id}")

        assets = doc.get("assets", {}) or {}
        kept = 0

        for asset_key, asset in assets.items():
            if not asset_is_science_relevant(asset):
                continue

            href = str(asset.get("href") or "")
            if not href:
                continue

            filename = safe_filename_from_href(href)
            scene_dir = dest_root / resolved_scene_id
            final_path = scene_dir / filename

            size = asset.get("file:size")
            if size is None:
                size = asset.get("file_size")

            rows.append(
                {
                    "original_scene_id": original_scene_id,
                    "resolved_scene_id": resolved_scene_id,
                    "l2a_time": r.get("l2a_time"),
                    "l2a_datatake_id": r.get("l2a_datatake_id"),
                    "tier": r.get("tier"),
                    "quality_class": r.get("quality_class"),
                    "supporting_records": r.get("supporting_records"),
                    "supporting_datasets_count": r.get(
                        "supporting_datasets_count"
                    ),
                    "supporting_datasets": r.get("supporting_datasets"),
                    "min_abs_delta_hours": r.get("min_abs_delta_hours"),
                    "asset_key": asset_key,
                    "asset_title": asset.get("title"),
                    "asset_roles": "|".join(asset.get("roles") or []),
                    "asset_type": asset.get("type"),
                    "url": href,
                    "expected_size_bytes": size,
                    "destination": str(final_path),
                }
            )
            kept += 1

        if kept == 0:
            print(
                f"  WARNING: no science assets resolved for "
                f"{resolved_scene_id}"
            )

    return rows


def save_df(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# DOWNLOAD
# =============================================================================

def _curl_config_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def curl_download(
    url: str,
    part_path: Path,
    username: str,
    password: str,
) -> Tuple[int, Optional[int]]:
    """
    Download with Basic auth. Progress stays visible on stderr; HTTP status is
    captured from stdout.
    """
    userpass = _curl_config_escape(f"{username}:{password}")
    url_escaped = _curl_config_escape(url)
    output_escaped = _curl_config_escape(str(part_path))

    config = "\n".join(
        [
            f'user = "{userpass}"',
            "basic",
            "location",
            "fail-with-body",
            "retry = 5",
            "retry-delay = 3",
            "retry-max-time = 600",
            "connect-timeout = 60",
            "speed-time = 120",
            "speed-limit = 1024",
            "continue-at = -",
            'write-out = "%{http_code}"',
            f'url = "{url_escaped}"',
            f'output = "{output_escaped}"',
            "",
        ]
    )

    result = subprocess.run(
        ["curl", "--config", "-"],
        input=config,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )

    text = (result.stdout or "").strip()
    m = re.search(r"(\d{3})\s*$", text)
    http_code = int(m.group(1)) if m else None
    return result.returncode, http_code


def refresh_one_asset(
    r: pd.Series,
    manifest_lookup: Dict[str, pd.Series],
    cache_dir: Path,
    dest_root: Path,
) -> Optional[Dict[str, Any]]:
    original_scene_id = str(r["original_scene_id"])
    src = manifest_lookup.get(original_scene_id)
    if src is None:
        return None

    doc = resolve_current_scene(
        src, cache_dir, force_refresh=True
    )
    resolved_scene_id = str(doc.get("id") or original_scene_id)
    asset_key = str(r["asset_key"])
    asset = (doc.get("assets", {}) or {}).get(asset_key)

    if not asset or not asset_is_science_relevant(asset):
        return None

    href = str(asset.get("href") or "")
    if not href:
        return None

    filename = safe_filename_from_href(href)
    final_path = dest_root / resolved_scene_id / filename

    return {
        "resolved_scene_id": resolved_scene_id,
        "url": href,
        "destination": str(final_path),
    }


def download_plan(
    plan: pd.DataFrame,
    manifest: pd.DataFrame,
    username: str,
    password: str,
    checkpoint_csv: Path,
    cache_dir: Path,
    dest_root: Path,
):
    total = len(plan)
    completed_rows: List[Dict[str, Any]] = []

    manifest_lookup = {
        str(r["l2a_scene_id"]).strip(): r
        for _, r in manifest.iterrows()
    }

    for i, (_, r0) in enumerate(plan.iterrows(), start=1):
        require_smb()

        r = r0.copy()
        final_path = Path(str(r["destination"]))
        part_path = Path(str(final_path) + ".part")
        final_path.parent.mkdir(parents=True, exist_ok=True)

        if final_path.exists() and final_path.stat().st_size > 0:
            print(f"[{i}/{total}] SKIP_EXISTS {final_path.name}")
            completed_rows.append(
                {**r.to_dict(), "download_status": "SKIP_EXISTS"}
            )
            continue

        print()
        print(f"[{i}/{total}] {r['resolved_scene_id']}")
        print(f"  asset: {r['asset_key']}")
        print(f"  file : {final_path.name}")

        # V2 may have left a tiny HTML 404 body in a .part file. Do not resume
        # from such a file. Real partial downloads larger than 1 MiB are kept.
        if part_path.exists() and part_path.stat().st_size < 1024 * 1024:
            print(
                f"  removing tiny stale .part "
                f"({part_path.stat().st_size} bytes)"
            )
            part_path.unlink()

        rc, http_code = curl_download(
            str(r["url"]), part_path, username, password
        )

        # A 404 is strongly suggestive of a product URL replaced during
        # reprocessing. Refresh current STAC product and retry this asset once.
        if rc != 0 and http_code == 404:
            print("  HTTP 404 -> refreshing CURRENT STAC product and retrying")

            # fail-with-body can leave a tiny HTML error body. Never resume it.
            if part_path.exists() and part_path.stat().st_size < 1024 * 1024:
                part_path.unlink()

            refreshed = refresh_one_asset(
                r,
                manifest_lookup=manifest_lookup,
                cache_dir=cache_dir,
                dest_root=dest_root,
            )

            if refreshed is not None:
                old_resolved = str(r["resolved_scene_id"])
                r["resolved_scene_id"] = refreshed["resolved_scene_id"]
                r["url"] = refreshed["url"]
                r["destination"] = refreshed["destination"]

                final_path = Path(str(r["destination"]))
                part_path = Path(str(final_path) + ".part")
                final_path.parent.mkdir(parents=True, exist_ok=True)

                if r["resolved_scene_id"] != old_resolved:
                    print(
                        f"  replacement product: "
                        f"{r['resolved_scene_id']}"
                    )

                rc, http_code = curl_download(
                    str(r["url"]), part_path, username, password
                )

        if rc != 0:
            if not smb_is_mounted():
                print()
                print("=" * 80)
                print("SMB DISCONNECTED DURING DOWNLOAD")
                print("=" * 80)
                print("Reconnect and rerun the SAME command.")
                save_df(checkpoint_csv, pd.DataFrame(completed_rows))
                raise SystemExit(21)

            # Delete tiny HTTP error pages, preserve genuine partial downloads.
            if (
                http_code is not None
                and http_code >= 400
                and part_path.exists()
                and part_path.stat().st_size < 1024 * 1024
            ):
                part_path.unlink()

            if http_code in {401, 403}:
                print(
                    f"  AUTHORIZATION_FAILED HTTP {http_code}. "
                    f"Check the doraaa/IPP credentials."
                )
                status = f"AUTH_FAILED_HTTP_{http_code}"
            elif http_code == 404:
                print(
                    "  STILL_404_AFTER_LIVE_REFRESH. "
                    "This scene needs catalogue review."
                )
                status = "STILL_404_AFTER_REFRESH"
            else:
                print(
                    f"  DOWNLOAD_FAILED curl_exit={rc} "
                    f"http={http_code}; real .part preserved"
                )
                status = f"DOWNLOAD_FAILED_{rc}_HTTP_{http_code}"

            completed_rows.append(
                {**r.to_dict(), "download_status": status}
            )
            save_df(checkpoint_csv, pd.DataFrame(completed_rows))
            raise SystemExit(22)

        if not part_path.exists() or part_path.stat().st_size <= 0:
            print("  ERROR: curl exited 0 but output is empty/missing")
            completed_rows.append(
                {**r.to_dict(), "download_status": "EMPTY_AFTER_DOWNLOAD"}
            )
            save_df(checkpoint_csv, pd.DataFrame(completed_rows))
            raise SystemExit(23)

        part_path.replace(final_path)
        print(
            f"  COMPLETE {final_path.stat().st_size / (1024**2):.1f} MB"
        )

        completed_rows.append(
            {**r.to_dict(), "download_status": "DOWNLOADED"}
        )

        if i % 25 == 0:
            save_df(checkpoint_csv, pd.DataFrame(completed_rows))
            print(f"  checkpoint -> {checkpoint_csv}")

    save_df(checkpoint_csv, pd.DataFrame(completed_rows))


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE))
    ap.add_argument("--username", default="doraaa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument(
        "--force-refresh-stac",
        action="store_true",
        help="Ignore V3 STAC cache and resolve all scenes live again.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    dest_root = Path(args.dest).expanduser()
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)

    if "quality_class" in manifest.columns:
        bad_q = (
            manifest["quality_class"]
            .astype(str)
            .str.upper()
            .ne("NOMINAL")
        )
        if bad_q.any():
            raise SystemExit(
                f"ABORT: manifest contains "
                f"{int(bad_q.sum())} non-NOMINAL scenes."
            )

    if "tier" in manifest.columns:
        bad_t = ~manifest["tier"].isin(["A_LE24H", "B_LE72H"])
        if bad_t.any():
            raise SystemExit(
                f"ABORT: manifest contains "
                f"{int(bad_t.sum())} non-primary tier scenes."
            )

    manifest = manifest.drop_duplicates("l2a_scene_id").copy()

    if args.limit and args.limit > 0:
        manifest = manifest.head(args.limit).copy()

    print("=" * 80)
    print("ENMAP PRIMARY <=72H NOMINAL DOWNLOADER V3 — FRESH RESOLUTION")
    print("=" * 80)
    print(f"Manifest       : {manifest_path}")
    print(f"Unique scenes  : {len(manifest)}")
    print(f"Destination    : {dest_root}")
    print(f"Username       : {args.username}")

    cache_dir = state_dir / "stac_current_items"
    plan_path = state_dir / (
        "download_plan_limit.csv" if args.limit
        else "download_plan_full.csv"
    )
    checkpoint = state_dir / (
        "download_checkpoint_limit.csv" if args.limit
        else "download_checkpoint_full.csv"
    )

    plan_rows = extract_plan_rows(
        manifest,
        cache_dir=cache_dir,
        dest_root=dest_root,
        force_refresh=args.force_refresh_stac,
    )
    plan = pd.DataFrame(plan_rows)
    save_df(plan_path, plan)

    original_n = manifest["l2a_scene_id"].nunique()
    resolved_n = (
        plan["resolved_scene_id"].nunique() if len(plan) else 0
    )

    replacements = 0
    if len(plan):
        scene_pairs = plan[
            ["original_scene_id", "resolved_scene_id"]
        ].drop_duplicates()
        replacements = int(
            (
                scene_pairs["original_scene_id"]
                != scene_pairs["resolved_scene_id"]
            ).sum()
        )

    known_sizes = pd.to_numeric(
        plan.get("expected_size_bytes"), errors="coerce"
    )
    known_n = int(known_sizes.notna().sum())
    known_gib = known_sizes.sum(skipna=True) / (1024**3)

    print()
    print("=" * 80)
    print("PREFLIGHT SUMMARY")
    print("=" * 80)
    print(f"Original manifest scenes : {original_n}")
    print(f"Current scenes resolved  : {resolved_n}")
    print(f"Scene IDs replaced       : {replacements}")
    print(f"Science asset files      : {len(plan)}")
    print(f"Assets with known size   : {known_n}")
    print(f"Known-size subtotal      : {known_gib:.2f} GiB")
    print(f"Download plan            : {plan_path}")

    if args.preflight_only or not args.download:
        print()
        print("No files downloaded.")
        return

    require_smb()

    if shutil.which("curl") is None:
        raise SystemExit("curl is required but was not found.")

    password = getpass.getpass(f"Password for {args.username}: ")
    if not password:
        raise SystemExit("Empty password; aborting.")

    print()
    print("=" * 80)
    print("DOWNLOAD START")
    print("=" * 80)

    download_plan(
        plan=plan,
        manifest=manifest,
        username=args.username,
        password=password,
        checkpoint_csv=checkpoint,
        cache_dir=cache_dir,
        dest_root=dest_root,
    )

    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(
        f"Current scenes: "
        f"{plan['resolved_scene_id'].nunique() if len(plan) else 0}"
    )
    print(f"Files         : {len(plan)}")
    print(f"Checkpoint    : {checkpoint}")


if __name__ == "__main__":
    main()
