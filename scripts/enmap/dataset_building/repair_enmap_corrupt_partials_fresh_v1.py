#!/usr/bin/env python3
"""
repair_enmap_corrupt_partials_fresh_v1.py

Safely repair EnMAP TIFF .part files that begin with a stale 1075-byte HTML
error body, using a FRESH EnMAP STAC lookup for the SAME exact acquisition.

Safety rules:
1) Fresh STAC query must resolve the SAME datatake + tile.
2) Fresh current_scene_id must equal the old current_scene_id.
   If DLR reprocessed/replaced the product, do NOT splice binaries.
3) Before changing byte 0, compare remote bytes against the local partial at:
     - immediately after the HTML prefix
     - near the current end of the partial (when large enough)
   Both windows must match exactly.
4) Only then fetch remote bytes 0..prefix_len-1, require TIFF magic,
   back up the contaminated prefix locally, and overwrite only that prefix.
5) File size must remain unchanged.

Default mode is DRY-RUN. Use --apply only after reviewing dry-run results.

Input:
  ~/methane_release_project/enmap_primary721_download_v2/
      download_checkpoint_full.csv

Output:
  ~/methane_release_project/enmap_corrupt_fresh_repair_v1/
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEMS = f"{STAC_ROOT}collections/{COLLECTION}/items"

DEFAULT_CHECKPOINT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary721_download_v2"
    / "download_checkpoint_full.csv"
)
DEFAULT_STATE = (
    Path.home()
    / "methane_release_project"
    / "enmap_corrupt_fresh_repair_v1"
)

USERNAME = "doraaa"
USER_AGENT = "UAlberta-EnMAP-Fresh-Corrupt-Repair/1.0"

SCENE_RE = re.compile(
    r"DT(?P<datatake>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


def is_tiff_magic(b: bytes) -> bool:
    return b[:4] in (
        b"II*\x00",
        b"MM\x00*",
        b"II+\x00",
        b"MM\x00+",
    )


def infer_html_prefix_len(part: Path) -> Optional[int]:
    with open(part, "rb") as f:
        head = f.read(65536)

    low = head.lower()
    if not low.startswith((b"<!doctype html", b"<html")):
        return None

    idx = low.find(b"</html>")
    if idx < 0:
        return None

    n = idx + len(b"</html>")
    for _ in range(4):
        if n < len(head) and head[n:n+1] in (b"\r", b"\n"):
            n += 1
        else:
            break
    return n


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
    d = re.sub(r"\D", "", str(value))
    return (d.lstrip("0") or "0") if d else None


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
            time.sleep(min(20, 2 ** (attempt - 1)))
    raise RuntimeError(f"STAC request failed: {url}\n{last}")


def fresh_exact_feature(scene_id: str) -> Optional[Dict[str, Any]]:
    parsed = parse_scene(scene_id)
    dt = normalize_digits(parsed.get("datatake"))
    tile = normalize_tile(parsed.get("tile"))
    acq = parsed.get("acq")

    if not dt or not tile or not acq:
        return None

    t = pd.to_datetime(
        acq,
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    if pd.isna(t):
        return None

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

    exact = [
        f
        for f in feats
        if feature_datatake(f) == dt
        and feature_tile(f) == tile
    ]

    if not exact:
        return None

    exact = sorted(
        exact,
        key=candidate_processing_key,
        reverse=True,
    )
    return exact[0]


def asset_href(feat: Dict[str, Any], asset_key: str) -> Optional[str]:
    assets = feat.get("assets", {}) or {}

    if asset_key in assets and assets[asset_key].get("href"):
        return str(assets[asset_key]["href"])

    wanted = str(asset_key).lower()

    for key, asset in assets.items():
        href = str(asset.get("href") or "")
        text = " ".join(
            [
                str(key),
                str(asset.get("title") or ""),
                href,
            ]
        ).lower()

        if wanted == "image" and "spectral_image" in text and href:
            return href

        if wanted in text and href:
            return href

    return None


def insert_scene_directory(url: str, scene_id: str) -> str:
    parts = urlsplit(url)
    pieces = parts.path.rstrip("/").split("/")
    filename = pieces[-1]

    if len(pieces) >= 2 and pieces[-2] == scene_id:
        return url

    new_path = "/".join(pieces[:-1] + [scene_id, filename])

    if parts.path.startswith("/") and not new_path.startswith("/"):
        new_path = "/" + new_path

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            new_path,
            parts.query,
            parts.fragment,
        )
    )


def curl_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def curl_range(
    url: str,
    username: str,
    password: str,
    start: int,
    end: int,
    max_time: int = 120,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Fetch exact remote byte range using curl. Returns body bytes.
    """
    userpass = curl_escape(f"{username}:{password}")
    url_e = curl_escape(url)

    config = "\n".join(
        [
            f'user = "{userpass}"',
            "basic",
            "location",
            "fail",
            "silent",
            "show-error",
            "connect-timeout = 30",
            f"max-time = {max_time}",
            f'range = "{start}-{end}"',
            f'url = "{url_e}"',
            "",
        ]
    ).encode()

    p = subprocess.run(
        ["curl", "--config", "-"],
        input=config,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if p.returncode != 0:
        return None, (
            f"curl_rc={p.returncode}; "
            f"stderr={p.stderr.decode('utf-8', errors='replace').strip()[:400]}"
        )

    expected = end - start + 1

    if len(p.stdout) != expected:
        return None, (
            f"length={len(p.stdout)}/{expected}; "
            f"stderr={p.stderr.decode('utf-8', errors='replace').strip()[:400]}"
        )

    return p.stdout, None


def working_fresh_url(
    href: str,
    scene_id: str,
    username: str,
    password: str,
    probe_start: int,
    probe_end: int,
):
    repaired = insert_scene_directory(href, scene_id)

    candidates = []
    seen = set()

    for style, url in [
        ("FRESH_ORIGINAL", href),
        ("FRESH_REPAIRED", repaired),
    ]:
        if url not in seen:
            seen.add(url)
            candidates.append((style, url))

    errors = []

    for style, url in candidates:
        data, err = curl_range(
            url,
            username,
            password,
            probe_start,
            probe_end,
        )

        if data is not None:
            return style, url, data, errors

        errors.append(f"{style}:{err}")

    return None, None, None, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state = Path(args.state_dir).expanduser().resolve()

    state.mkdir(parents=True, exist_ok=True)
    backups = state / "prefix_backups"
    backups.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    df = pd.read_csv(checkpoint, low_memory=False)

    bad = df[
        df["download_status"]
        .fillna("")
        .astype(str)
        .str.startswith("CORRUPT_PART_PREFIX:")
    ].copy()

    if args.limit > 0:
        bad = bad.head(args.limit).copy()

    print("=" * 96)
    print("ENMAP CORRUPT PARTIAL REPAIR — FRESH EXACT HREF + BINARY VERIFY")
    print("=" * 96)
    print("Rows :", len(bad))
    print("Mode :", "APPLY" if args.apply else "DRY-RUN")
    print()

    password = getpass.getpass(f"IPS password for {USERNAME}: ")
    if not password:
        raise SystemExit("Empty password; aborting.")

    feature_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    results = []

    for pos, (_, r) in enumerate(bad.iterrows(), start=1):
        old_current = str(r["current_scene_id"]).strip()
        asset_key = str(r["asset_key"]).strip()
        final_path = Path(str(r["destination"]))
        part = Path(str(final_path) + ".part")

        base = {
            "old_current_scene_id": old_current,
            "asset_key": asset_key,
            "part_path": str(part),
            "part_size_bytes": (
                part.stat().st_size
                if part.exists()
                else 0
            ),
        }

        print(f"[{pos}/{len(bad)}] {old_current} :: {asset_key}")

        if not part.exists():
            print("  MISSING_PART")
            results.append({**base, "status": "MISSING_PART"})
            continue

        if final_path.suffix.lower() not in (".tif", ".tiff"):
            print("  NOT_TIFF_PART")
            results.append({**base, "status": "NOT_TIFF_PART"})
            continue

        n = infer_html_prefix_len(part)

        if not n:
            with open(part, "rb") as f:
                head = f.read(16)

            if is_tiff_magic(head):
                print("  ALREADY_TIFF")
                results.append({**base, "status": "ALREADY_TIFF"})
            else:
                print("  HTML_PREFIX_NOT_IDENTIFIED")
                results.append({
                    **base,
                    "status": "HTML_PREFIX_NOT_IDENTIFIED",
                    "first16": repr(head),
                })
            continue

        if old_current not in feature_cache:
            try:
                feature_cache[old_current] = fresh_exact_feature(old_current)
            except Exception as e:
                feature_cache[old_current] = None
                print(f"  STAC_ERROR: {type(e).__name__}: {e}")

        feat = feature_cache.get(old_current)

        if feat is None:
            print("  NO_FRESH_EXACT_PRODUCT")
            results.append({
                **base,
                "status": "NO_FRESH_EXACT_PRODUCT",
                "prefix_len": n,
            })
            continue

        fresh_id = str(feat.get("id") or "")

        if fresh_id != old_current:
            print(f"  PRODUCT_CHANGED -> {fresh_id}")
            results.append({
                **base,
                "status": "PRODUCT_CHANGED",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
            })
            continue

        href = asset_href(feat, asset_key)

        if not href:
            print("  FRESH_ASSET_HREF_MISSING")
            results.append({
                **base,
                "status": "FRESH_ASSET_HREF_MISSING",
                "prefix_len": n,
            })
            continue

        part_size = part.stat().st_size
        win = int(args.window)

        if part_size <= n:
            print("  PART_HAS_NO_TIFF_TAIL")
            results.append({
                **base,
                "status": "PART_HAS_NO_TIFF_TAIL",
                "prefix_len": n,
            })
            continue

        # First overlap immediately after contaminated prefix.
        start1 = n
        end1 = min(part_size - 1, n + win - 1)

        style, good_url, remote1, errors = working_fresh_url(
            href,
            fresh_id,
            USERNAME,
            password,
            start1,
            end1,
        )

        if remote1 is None:
            print("  FRESH_URL_NOT_DOWNLOADABLE")
            results.append({
                **base,
                "status": "FRESH_URL_NOT_DOWNLOADABLE",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "errors": " | ".join(errors),
            })
            continue

        with open(part, "rb") as f:
            f.seek(start1)
            local1 = f.read(end1 - start1 + 1)

        if local1 != remote1:
            print("  BINARY_MISMATCH_AFTER_PREFIX")
            results.append({
                **base,
                "status": "BINARY_MISMATCH_AFTER_PREFIX",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "working_url_style": style,
            })
            continue

        # Second overlap near current end, when enough data exists.
        tail_match = True
        start2 = None
        end2 = None

        if part_size >= n + 2 * win:
            start2 = max(n, part_size - win)
            end2 = part_size - 1

            remote2, err2 = curl_range(
                good_url,
                USERNAME,
                password,
                start2,
                end2,
            )

            if remote2 is None:
                print(f"  TAIL_PROBE_FAILED: {err2}")
                results.append({
                    **base,
                    "status": "TAIL_PROBE_FAILED",
                    "prefix_len": n,
                    "fresh_current_scene_id": fresh_id,
                    "working_url_style": style,
                    "error": err2,
                })
                continue

            with open(part, "rb") as f:
                f.seek(start2)
                local2 = f.read(end2 - start2 + 1)

            tail_match = local2 == remote2

            if not tail_match:
                print("  BINARY_MISMATCH_NEAR_TAIL")
                results.append({
                    **base,
                    "status": "BINARY_MISMATCH_NEAR_TAIL",
                    "prefix_len": n,
                    "fresh_current_scene_id": fresh_id,
                    "working_url_style": style,
                })
                continue

        # Fetch the correct prefix only after two-window binary continuity.
        prefix, prefix_err = curl_range(
            good_url,
            USERNAME,
            password,
            0,
            n - 1,
        )

        if prefix is None:
            print(f"  PREFIX_FETCH_FAILED: {prefix_err}")
            results.append({
                **base,
                "status": "PREFIX_FETCH_FAILED",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "working_url_style": style,
                "error": prefix_err,
            })
            continue

        if not is_tiff_magic(prefix):
            print(f"  PREFIX_NOT_TIFF: {prefix[:16]!r}")
            results.append({
                **base,
                "status": "PREFIX_NOT_TIFF",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "working_url_style": style,
                "prefix_first16": repr(prefix[:16]),
            })
            continue

        print(
            f"  VERIFIED: same product; "
            f"{style}; prefix_len={n}; "
            f"overlap1=OK; tail=OK"
        )

        if not args.apply:
            print("  WOULD_REPAIR")
            results.append({
                **base,
                "status": "WOULD_REPAIR",
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "working_url_style": style,
                "verify_start1": start1,
                "verify_end1": end1,
                "verify_start2": start2,
                "verify_end2": end2,
            })
        else:
            safe_asset = "".join(
                c if c.isalnum() or c in "-_."
                else "_"
                for c in asset_key
            )

            backup = (
                backups
                / f"{fresh_id}__{safe_asset}__{n}bytes.bin"
            )

            with open(part, "rb") as f:
                old_prefix = f.read(n)

            if not backup.exists():
                backup.write_bytes(old_prefix)

            size_before = part.stat().st_size

            with open(part, "r+b") as f:
                f.seek(0)
                f.write(prefix)
                f.flush()

            size_after = part.stat().st_size

            with open(part, "rb") as f:
                verify_head = f.read(16)

            if (
                size_before == size_after
                and is_tiff_magic(verify_head)
            ):
                status = "REPAIRED"
                print("  ✅ REPAIRED")
            else:
                status = "REPAIR_VERIFY_FAILED"
                print("  REPAIR_VERIFY_FAILED")

            results.append({
                **base,
                "status": status,
                "prefix_len": n,
                "fresh_current_scene_id": fresh_id,
                "working_url_style": style,
                "backup_path": str(backup),
                "size_before": size_before,
                "size_after": size_after,
            })

        if pos % 25 == 0:
            cp = state / "repair_checkpoint.csv"
            pd.DataFrame(results).to_csv(cp, index=False)
            print(f"  checkpoint -> {cp}")

        if pos < len(bad):
            time.sleep(args.sleep)

    out = pd.DataFrame(results)

    out_path = state / (
        "repair_results_apply.csv"
        if args.apply
        else "repair_results_dryrun.csv"
    )
    out.to_csv(out_path, index=False)

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    if len(out):
        print(out["status"].value_counts(dropna=False).to_string())

    if "working_url_style" in out.columns:
        print()
        print("WORKING URL STYLES")
        print(
            out["working_url_style"]
            .dropna()
            .value_counts()
            .to_string()
        )

    if "prefix_len" in out.columns:
        print()
        print("PREFIX LENGTHS")
        print(
            out["prefix_len"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
