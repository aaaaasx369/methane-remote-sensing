#!/usr/bin/env python3
"""
download_enmap_primary721_v4_safe_partials.py

Download the CONFIRMED-DOWNLOADABLE EnMAP PRIMARY dataset:
    <=72 h + NOMINAL L2A
using the updated 721-scene availability manifest.

Default input:
  ~/methane_release_project/enmap_primary762_availability_retry_v1/
      03_downloadable_scenes_updated.csv

Default destination:
  /Volumes/engg-leung/dora lin/EnMAP_MethaneFuse/
      01_raw_L2A/primary72h_nominal/

Design
------
- Uses current_scene_id from the frozen availability audit.
- Fetches the CURRENT STAC item directly by item ID (no time-window search).
- Validates original datatake + tile against the current item.
- Downloads science assets only:
    spectral image + metadata + quality/mask layers
  and excludes quicklooks/thumbnails.
- For EACH asset independently:
    original STAC href first;
    if HTTP 404, try repaired /<SCENE_ID>/ path.
- Direct to lab SMB; no staging on Mac.
- .part resume with curl.
- V2 automatically reopens and resumes curl 18/56 transport interruptions
  up to 25 sessions per asset.
- Existing non-empty completed files are skipped.
- SMB disconnect => safe stop.
- 401/403 => abort immediately.
- Other per-asset failures are logged and the run continues.
- Local STAC cache + checkpoints survive reruns.
- Password is requested interactively and never written to disk.

Recommended:
  # 5-scene smoke test
  caffeinate -i python3 download_enmap_primary721_v4_safe_partials.py \
      --download --limit 5 --username doraaa

  # full 721-scene run
  caffeinate -i python3 download_enmap_primary721_v4_safe_partials.py \
      --download --username doraaa
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import urllib3


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEM_BASE = f"{STAC_ROOT}collections/{COLLECTION}/items/"

DEFAULT_PROJECT = Path.home() / "methane_release_project"
DEFAULT_MANIFEST = (
    DEFAULT_PROJECT
    / "enmap_primary762_availability_retry_v1"
    / "03_downloadable_scenes_updated.csv"
)
DEFAULT_DEST = Path(
    "/Volumes/engg-leung/dora lin/EnMAP_MethaneFuse/"
    "01_raw_L2A/primary72h_nominal"
)
DEFAULT_STATE = DEFAULT_PROJECT / "enmap_primary721_download_v2"

SMB_MOUNT = Path("/Volumes/engg-leung")
USER_AGENT = "UAlberta-EnMAP-Primary721-Downloader/4.0"

EXCLUDE_ROLES = {"overview", "thumbnail"}

SCENE_RE = re.compile(
    r"DT(?P<datatake>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


# =============================================================================
# Basic helpers
# =============================================================================

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


def exact_same_acquisition(
    feat: Dict[str, Any],
    original_scene_id: str,
) -> bool:
    p = parse_scene(original_scene_id)
    wanted_dt = normalize_digits(p.get("datatake"))
    wanted_tile = normalize_tile(p.get("tile"))
    return (
        wanted_dt is not None
        and wanted_tile is not None
        and feature_datatake(feat) == wanted_dt
        and feature_tile(feat) == wanted_tile
    )


def insert_scene_directory(url: str, scene_id: str) -> str:
    parts = urlsplit(url)
    pieces = parts.path.rstrip("/").split("/")
    if not pieces:
        return url

    filename = pieces[-1]

    if len(pieces) >= 2 and pieces[-2] == scene_id:
        return url

    new_path = "/".join(pieces[:-1] + [scene_id, filename])
    if parts.path.startswith("/") and not new_path.startswith("/"):
        new_path = "/" + new_path

    return urlunsplit(
        (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    )


def safe_filename_from_href(href: str) -> str:
    return href.split("?")[0].rstrip("/").split("/")[-1]


# =============================================================================
# SMB
# =============================================================================

def smb_is_mounted() -> bool:
    try:
        r = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f" on {SMB_MOUNT} " in r.stdout
    except Exception:
        return False


def require_smb():
    if not smb_is_mounted():
        print()
        print("=" * 88)
        print("SMB SHARE IS NOT MOUNTED")
        print("=" * 88)
        print(f"Expected: {SMB_MOUNT}")
        print("Reconnect the share, then rerun the SAME command.")
        print("Completed files and genuine .part files are preserved.")
        raise SystemExit(20)


# =============================================================================
# STAC direct-item cache
# =============================================================================

def fetch_current_item(
    http: urllib3.PoolManager,
    current_scene_id: str,
    cache_dir: Path,
    retries: int = 8,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{current_scene_id}.json"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = (
        STAC_ITEM_BASE
        + quote(current_scene_id, safe="")
        + "?f=json"
    )

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = http.request(
                "GET",
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/geo+json, application/json",
                },
                preload_content=True,
                redirect=True,
            )

            if r.status == 200:
                doc = json.loads(r.data.decode("utf-8"))
                tmp = cache_path.with_suffix(".json.part")
                tmp.write_text(
                    json.dumps(doc, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(cache_path)
                return doc

            last_error = f"HTTP {r.status}"

            if r.status in (401, 403, 404):
                raise RuntimeError(
                    f"Direct STAC item HTTP {r.status}: {url}"
                )

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries:
            wait = min(90.0, 3.0 * (2 ** (attempt - 1)))
            print(
                f"  STAC retry {attempt}/{retries}: "
                f"{last_error}; sleep {wait:.1f}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch direct STAC item "
        f"{current_scene_id}: {last_error}"
    )


# =============================================================================
# Asset selection
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

    text = " ".join([
        str(asset.get("title") or ""),
        str(asset.get("description") or ""),
        href,
    ]).upper()

    keep_tokens = [
        "SPECTRAL_IMAGE",
        "METADATA",
        "QUALITY",
        "PIXELMASK",
        "DEFECTIVE",
    ]

    return any(tok in text for tok in keep_tokens)


def build_scene_assets(
    row: pd.Series,
    doc: Dict[str, Any],
    dest_root: Path,
) -> List[Dict[str, Any]]:
    original_scene_id = str(row["original_scene_id"]).strip()
    current_scene_id = str(row["current_scene_id"]).strip()

    if str(doc.get("id") or "") != current_scene_id:
        raise RuntimeError(
            f"STAC item ID mismatch: expected {current_scene_id}, "
            f"got {doc.get('id')}"
        )

    if not exact_same_acquisition(doc, original_scene_id):
        raise RuntimeError(
            f"Exact datatake+tile validation failed for "
            f"{original_scene_id} -> {current_scene_id}"
        )

    out = []

    for asset_key, asset in (doc.get("assets", {}) or {}).items():
        if not asset_is_science_relevant(asset):
            continue

        original_url = str(asset.get("href") or "")
        repaired_url = insert_scene_directory(
            original_url,
            current_scene_id,
        )

        filename = safe_filename_from_href(original_url)

        out.append({
            "original_scene_id": original_scene_id,
            "current_scene_id": current_scene_id,
            "tier": row.get("tier"),
            "supporting_datasets": row.get("supporting_datasets"),
            "min_abs_delta_hours": row.get("min_abs_delta_hours"),
            "asset_key": asset_key,
            "asset_title": asset.get("title"),
            "asset_roles": "|".join(asset.get("roles") or []),
            "asset_type": asset.get("type"),
            "availability_status": row.get("status"),
            "availability_path_style": row.get("path_style"),
            "original_url": original_url,
            "repaired_url": repaired_url,
            "destination": str(
                dest_root / current_scene_id / filename
            ),
        })

    return out


# =============================================================================
# curl downloader
# =============================================================================

def _curl_config_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def curl_download_once(
    url: str,
    part_path: Path,
    username: str,
    password: str,
) -> Tuple[int, Optional[int]]:
    """One curl transfer session, always resuming from the existing .part."""
    userpass = _curl_config_escape(
        f"{username}:{password}"
    )
    url_escaped = _curl_config_escape(url)
    output_escaped = _curl_config_escape(str(part_path))

    config = "\n".join([
        f'user = "{userpass}"',
        "basic",
        "location",
        "fail",
        "retry = 6",
        "retry-delay = 3",
        "retry-max-time = 900",
        "connect-timeout = 60",
        "speed-time = 180",
        "speed-limit = 1024",
        "continue-at = -",
        'write-out = "%{http_code}"',
        f'url = "{url_escaped}"',
        f'output = "{output_escaped}"',
        "",
    ])

    result = subprocess.run(
        ["curl", "--config", "-"],
        input=config,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )

    out_text = (result.stdout or "").strip()
    m = re.search(r"(\d{3})\s*$", out_text)
    http_code = int(m.group(1)) if m else None

    return result.returncode, http_code


def curl_download_resilient(
    url: str,
    part_path: Path,
    username: str,
    password: str,
    max_sessions: int = 25,
) -> Tuple[int, Optional[int], int]:
    """Repeatedly reopen interrupted range transfers and resume .part."""
    retryable_transport = {18, 28, 35, 52, 55, 56, 92}
    last_rc = 0
    last_http = None

    for session in range(1, max_sessions + 1):
        before = part_path.stat().st_size if part_path.exists() else 0
        rc, http_code = curl_download_once(
            url, part_path, username, password
        )
        last_rc, last_http = rc, http_code
        after = part_path.stat().st_size if part_path.exists() else 0

        if rc == 0 and http_code in (200, 206):
            return rc, http_code, session

        if http_code in (401, 403, 404):
            return rc, http_code, session

        if rc in retryable_transport and session < max_sessions:
            gained = max(0, after - before)
            print(
                f"    transport interruption: curl={rc} http={http_code}; "
                f"part={after/(1024**2):.1f} MB "
                f"(+{gained/(1024**2):.1f} MB this session)"
            )
            wait = min(30, 2 + session * 2)
            print(
                f"    resume session {session+1}/{max_sessions} after {wait}s"
            )
            time.sleep(wait)
            continue

        return rc, http_code, session

    return last_rc, last_http, max_sessions


def clean_tiny_http_error_part(
    part_path: Path,
    http_code: Optional[int],
):
    if (
        http_code is not None
        and http_code >= 400
        and part_path.exists()
        and part_path.stat().st_size < 1024 * 1024
    ):
        part_path.unlink()


def validate_partial_prefix(
    part_path: Path,
    final_path: Path,
) -> Tuple[bool, str]:
    """
    Validate byte 0 before Range-resuming an existing partial.

    This prevents an old HTML/HTTP error body from becoming the beginning of
    a TIFF/XML file.
    """
    if not part_path.exists() or part_path.stat().st_size == 0:
        return True, "EMPTY_OR_MISSING"

    with open(part_path, "rb") as f:
        head = f.read(64)

    name = final_path.name.lower()

    if name.endswith((".tif", ".tiff")):
        ok = head[:4] in (
            b"II*\x00",
            b"MM\x00*",
            b"II+\x00",
            b"MM\x00+",
        )
        return ok, "TIFF" if ok else f"BAD_TIFF_PREFIX:{head[:16]!r}"

    if name.endswith(".xml"):
        stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
        ok = stripped.startswith(b"<")
        return ok, "XML" if ok else f"BAD_XML_PREFIX:{head[:16]!r}"

    return True, "UNVALIDATED_EXTENSION"



def download_one_asset(
    row: pd.Series,
    username: str,
    password: str,
) -> Dict[str, Any]:
    """
    Download one asset safely.

    Critical V3 fix:
    A genuine .part may have been created from the repaired /<SCENE_ID>/ URL.
    V2 always tried the original STAC URL first on rerun, which could return
    403/404 to a Range request and make curl report code 33. V3 uses the
    scene-level availability path style to choose the likely-correct URL first
    whenever a partial file already exists, and it NEVER treats a resume-only
    403 as an authentication failure until the alternate path has been tried.
    """
    require_smb()

    final_path = Path(str(row["destination"]))
    part_path = Path(str(final_path) + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and final_path.stat().st_size > 0:
        return {
            **row.to_dict(),
            "download_status": "SKIP_EXISTS",
            "used_url_style": "EXISTING",
            "final_size_bytes": final_path.stat().st_size,
            "http_code": None,
            "resume_sessions": 0,
        }

    # Remove only tiny stale HTTP error bodies.
    if part_path.exists() and part_path.stat().st_size < 1024:
        part_path.unlink()

    # Critical V4 guard: never resume a TIFF/XML partial whose byte 0 is
    # already contaminated by an HTML/HTTP error response.
    ok_prefix, prefix_note = validate_partial_prefix(
        part_path, final_path
    )
    if not ok_prefix:
        return {
            **row.to_dict(),
            "download_status": f"CORRUPT_PART_PREFIX:{prefix_note}",
            "used_url_style": None,
            "final_size_bytes": (
                part_path.stat().st_size
                if part_path.exists()
                else 0
            ),
            "http_code": None,
            "resume_sessions": 0,
        }

    original_url = str(row["original_url"])
    repaired_url = str(row["repaired_url"])
    path_style = str(row.get("availability_path_style") or "")

    had_real_partial = (
        part_path.exists()
        and part_path.stat().st_size >= 1024
    )

    # For a real partial, prefer the path style already proven by the
    # availability audit. This is the key V3 behavior.
    if path_style == "SCENE_DIRECTORY_INSERTED":
        candidates = [
            ("REPAIRED", repaired_url),
            ("ORIGINAL", original_url),
        ]
    elif path_style == "ORIGINAL":
        candidates = [
            ("ORIGINAL", original_url),
            ("REPAIRED", repaired_url),
        ]
    else:
        # Conservative fallback. Existing partials are more often associated
        # with the repaired path in this archive; fresh downloads retain the
        # traditional original-first ordering.
        candidates = (
            [("REPAIRED", repaired_url), ("ORIGINAL", original_url)]
            if had_real_partial
            else [("ORIGINAL", original_url), ("REPAIRED", repaired_url)]
        )

    # De-duplicate identical URLs.
    seen = set()
    ordered = []
    for style, url in candidates:
        if url not in seen:
            seen.add(url)
            ordered.append((style, url))

    last_rc = None
    last_http = None
    total_sessions = 0
    used_style = None

    for candidate_index, (style, url) in enumerate(ordered, start=1):
        before = part_path.stat().st_size if part_path.exists() else 0

        if before > 0:
            print(
                f"  resume candidate {candidate_index}/{len(ordered)}: "
                f"{style} from {before} bytes"
            )
        else:
            print(
                f"  URL candidate {candidate_index}/{len(ordered)}: {style}"
            )

        rc, http_code, sessions = curl_download_resilient(
            url,
            part_path,
            username,
            password,
            max_sessions=25,
        )
        total_sessions += sessions
        last_rc = rc
        last_http = http_code

        if not smb_is_mounted():
            raise RuntimeError("SMB_DISCONNECTED")

        if rc == 0 and http_code in (200, 206):
            used_style = style
            break

        # curl 33 means the resume attempt was rejected as non-range-capable.
        # A 403/404 on a request that is resuming a real partial can also be a
        # URL-layout/range issue rather than bad credentials. Preserve the
        # partial and try the alternate URL.
        current_part = (
            part_path.stat().st_size
            if part_path.exists()
            else 0
        )
        resume_context = had_real_partial or current_part >= 1024

        if rc == 33:
            print(
                f"  {style} cannot resume this partial (curl 33); "
                "preserving .part and trying alternate path"
            )
            continue

        if http_code in (403, 404) and resume_context:
            print(
                f"  {style} returned HTTP {http_code} during resume; "
                "preserving .part and trying alternate path"
            )
            continue

        # Fresh 404: normal path-layout miss; try alternate URL.
        if http_code == 404:
            clean_tiny_http_error_part(part_path, http_code)
            continue

        # Fresh 401/403 after no genuine partial context is a real auth failure.
        if http_code in (401, 403):
            raise PermissionError(
                f"AUTH_FAILED_HTTP_{http_code}"
            )

        # Other transfer failures: alternate path usually will not help.
        # Leave the genuine partial intact for the next rerun.
        break

    if used_style is not None:
        if not part_path.exists() or part_path.stat().st_size <= 0:
            return {
                **row.to_dict(),
                "download_status": "EMPTY_AFTER_DOWNLOAD",
                "used_url_style": used_style,
                "final_size_bytes": 0,
                "http_code": last_http,
                "resume_sessions": total_sessions,
            }

        part_path.replace(final_path)

        return {
            **row.to_dict(),
            "download_status": "DOWNLOADED",
            "used_url_style": used_style,
            "final_size_bytes": final_path.stat().st_size,
            "http_code": last_http,
            "resume_sessions": total_sessions,
        }

    return {
        **row.to_dict(),
        "download_status": (
            f"FAILED_CURL_{last_rc}_HTTP_{last_http}"
        ),
        "used_url_style": None,
        "final_size_bytes": (
            part_path.stat().st_size
            if part_path.exists()
            else 0
        ),
        "http_code": last_http,
        "resume_sessions": total_sessions,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    ap.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
    )
    ap.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE),
    )
    ap.add_argument(
        "--username",
        default="doraaa",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all scenes",
    )
    ap.add_argument(
        "--download",
        action="store_true",
    )
    args = ap.parse_args()

    manifest_path = Path(
        args.manifest
    ).expanduser().resolve()

    dest_root = Path(
        args.dest
    ).expanduser()

    state_dir = Path(
        args.state_dir
    ).expanduser().resolve()

    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    manifest = manifest.drop_duplicates(
        "original_scene_id"
    ).copy()

    if args.limit > 0:
        manifest = manifest.head(
            args.limit
        ).copy()

    print("=" * 88)
    print("ENMAP PRIMARY 721 DOWNLOADER V4 — SAFE PARTIALS")
    print("=" * 88)
    print(f"Manifest      : {manifest_path}")
    print(f"Unique scenes : {len(manifest)}")
    print(f"Destination   : {dest_root}")
    print(f"Username      : {args.username}")
    print()

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=45.0,
            read=120.0,
        ),
        retries=False,
        num_pools=4,
    )

    cache_dir = state_dir / "stac_items"

    plan_rows: List[Dict[str, Any]] = []
    scene_errors: List[Dict[str, Any]] = []

    for pos, (_, r) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        original_scene_id = str(
            r["original_scene_id"]
        ).strip()
        current_scene_id = str(
            r["current_scene_id"]
        ).strip()

        print(
            f"[resolve {pos}/{len(manifest)}] "
            f"{current_scene_id}"
        )

        try:
            doc = fetch_current_item(
                http,
                current_scene_id,
                cache_dir,
            )
            assets = build_scene_assets(
                r,
                doc,
                dest_root,
            )

            if not assets:
                raise RuntimeError(
                    "No science assets found"
                )

            plan_rows.extend(assets)

        except Exception as e:
            print(
                f"  RESOLVE_ERROR: "
                f"{type(e).__name__}: {e}"
            )
            scene_errors.append({
                "original_scene_id": original_scene_id,
                "current_scene_id": current_scene_id,
                "error": f"{type(e).__name__}: {e}",
            })

    plan = pd.DataFrame(plan_rows)

    plan_path = state_dir / (
        "download_plan_limit.csv"
        if args.limit
        else "download_plan_full.csv"
    )
    plan.to_csv(plan_path, index=False)

    errors_path = state_dir / (
        "resolve_errors_limit.csv"
        if args.limit
        else "resolve_errors_full.csv"
    )
    pd.DataFrame(scene_errors).to_csv(
        errors_path,
        index=False,
    )

    print()
    print("=" * 88)
    print("PLAN SUMMARY")
    print("=" * 88)
    print(
        f"Scenes planned    : "
        f"{plan['current_scene_id'].nunique() if len(plan) else 0}"
    )
    print(
        f"Science assets    : {len(plan)}"
    )
    print(
        f"Resolve errors    : {len(scene_errors)}"
    )
    print(f"Plan              : {plan_path}")

    if not args.download:
        print()
        print("No files downloaded.")
        print("Use --download to begin.")
        return

    require_smb()

    if shutil.which("curl") is None:
        raise SystemExit(
            "curl is required but was not found."
        )

    password = getpass.getpass(
        f"IPS password for {args.username}: "
    )

    if not password:
        raise SystemExit(
            "Empty password; aborting."
        )

    checkpoint_path = state_dir / (
        "download_checkpoint_limit.csv"
        if args.limit
        else "download_checkpoint_full.csv"
    )

    completed: List[Dict[str, Any]] = []

    print()
    print("=" * 88)
    print("DOWNLOAD START")
    print("=" * 88)

    for pos, (_, r) in enumerate(
        plan.iterrows(),
        start=1,
    ):
        print()
        print(
            f"[{pos}/{len(plan)}] "
            f"{r['current_scene_id']}"
        )
        print(
            f"  asset: {r['asset_key']}"
        )
        print(
            f"  file : "
            f"{Path(str(r['destination'])).name}"
        )

        try:
            result = download_one_asset(
                r,
                args.username,
                password,
            )

        except PermissionError as e:
            print(
                f"  AUTHORIZATION FAILURE: {e}"
            )
            pd.DataFrame(completed).to_csv(
                checkpoint_path,
                index=False,
            )
            raise SystemExit(30)

        except RuntimeError as e:
            if str(e) == "SMB_DISCONNECTED":
                print()
                print("=" * 88)
                print("SMB DISCONNECTED")
                print("=" * 88)
                print(
                    "Reconnect and rerun the SAME command."
                )
                pd.DataFrame(completed).to_csv(
                    checkpoint_path,
                    index=False,
                )
                raise SystemExit(31)

            result = {
                **r.to_dict(),
                "download_status": (
                    f"RUNTIME_ERROR:{e}"
                ),
                "used_url_style": None,
                "final_size_bytes": 0,
                "http_code": None,
            }

        status = result["download_status"]

        if status in (
            "DOWNLOADED",
            "SKIP_EXISTS",
        ):
            mb = (
                float(result.get(
                    "final_size_bytes", 0
                ))
                / (1024 ** 2)
            )
            print(
                f"  {status} {mb:.1f} MB "
                f"via {result.get('used_url_style')}"
            )
        else:
            print(
                f"  {status}"
            )

        completed.append(result)

        if pos % 25 == 0:
            pd.DataFrame(completed).to_csv(
                checkpoint_path,
                index=False,
            )
            print(
                f"  checkpoint -> "
                f"{checkpoint_path}"
            )

    done = pd.DataFrame(completed)
    done.to_csv(
        checkpoint_path,
        index=False,
    )

    counts = done[
        "download_status"
    ].value_counts(dropna=False)

    total_bytes = pd.to_numeric(
        done.get("final_size_bytes"),
        errors="coerce",
    ).fillna(0).sum()

    print()
    print("=" * 88)
    print("DOWNLOAD RUN COMPLETE")
    print("=" * 88)
    print(
        f"Scenes in plan : "
        f"{plan['current_scene_id'].nunique()}"
    )
    print(
        f"Assets in plan : {len(plan)}"
    )
    print(
        f"Completed size : "
        f"{total_bytes / (1024**3):.2f} GiB"
    )
    print()
    print("STATUS COUNTS")
    print(counts.to_string())
    print()
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
