#!/usr/bin/env python3
"""
Stanford 2025 EnMAP downloader v4 — authenticated DLR Geoservice download.

The DLR EnMAP STAC metadata are discoverable, but the actual EnMAP L2A file
downloads require a valid EnMAP/EOC Geoservice user account. DLR's official
example uses HTTP Basic Authentication on asset requests.

Features
--------
- exact EnMAP L2A product IDs from the Stanford download master
- fetches live DLR STAC item for authoritative asset hrefs
- authenticates asset downloads with HTTP Basic Auth
- username via --username / ENMAP_USERNAME / interactive prompt
- password via ENMAP_PASSWORD / hidden getpass prompt
- validates TIFF/JPEG/XML content; rejects HTML/SSO login pages
- retries transient failures
- resumes .part files with HTTP Range
- resets stale .part on HTTP 416
- SMB-safe wait/resume for /Volumes/engg-leung
- keeps existing valid files, repairs the earlier HTML masquerading as .TIF

Default output:
  /Volumes/engg-leung/dora lin/Stanford_2025_AllSensor/EnMAP
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

DEFAULT_ROOT = "/Volumes/engg-leung/dora lin/Stanford_2025_AllSensor"
DEFAULT_MOUNT = "/Volumes/engg-leung"
STAC_BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION = "ENMAP_HSI_L2A"
CHUNK = 8 * 1024 * 1024


class AuthenticationFailure(RuntimeError):
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download-master", required=True)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--mount", default=DEFAULT_MOUNT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--wait-seconds", type=int, default=30)
    p.add_argument("--username", default=os.environ.get("ENMAP_USERNAME", ""))
    return p.parse_args()


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._+\-]+', "_", str(s)).strip("_")


def filename_from_url(url, fallback="asset.bin"):
    name = Path(unquote(urlparse(url).path)).name
    return safe_name(name or fallback)


def mount_active(mount: Path):
    try:
        return mount.exists() and os.path.ismount(mount)
    except OSError:
        return False


def wait_for_mount(mount: Path, wait_seconds: int):
    announced = False
    while not mount_active(mount):
        if not announced:
            print()
            print("=" * 80)
            print("SMB SHARE DISCONNECTED — WAITING")
            print("=" * 80)
            print("Reconnect:")
            print("  open 'smb://smb.research-filer.ualberta.ca/engg-leung'")
            print()
            print("Canonical mount must be /Volumes/engg-leung")
            print("not /Volumes/engg-leung-1")
            print(f"Checking every {wait_seconds} seconds...")
            announced = True
        time.sleep(wait_seconds)
    if announced:
        print("✅ SMB mount is back. Resuming.")
        print()


def ensure_dirs(root: Path, mount: Path, wait_seconds: int):
    wait_for_mount(mount, wait_seconds)
    root.mkdir(parents=True, exist_ok=True)
    (root / "EnMAP").mkdir(exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    test = root / "manifests" / ".enmap_v4_write_test"
    test.write_text("ok", encoding="utf-8")
    test.unlink()


def append_log(log_path: Path, mount: Path, wait_seconds: int, row: dict):
    fields = [
        "timestamp_utc", "sensor", "release_ID", "product_family",
        "product_id", "asset", "status", "bytes", "path", "error"
    ]

    while True:
        wait_for_mount(mount, wait_seconds)
        try:
            exists = log_path.exists()
            with log_path.open("a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    w.writeheader()
                out = {k: row.get(k, "") for k in fields}
                out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                w.writerow(out)
            return
        except OSError as e:
            print(f"  manifest write interrupted: {e!r}")
            time.sleep(wait_seconds)


def read_prefix(path: Path, n=4096):
    try:
        with path.open("rb") as f:
            return f.read(n)
    except Exception:
        return b""


def looks_like_html(prefix: bytes):
    low = prefix.lstrip().lower()
    return (
        low.startswith(b"<!doctype html")
        or low.startswith(b"<html")
        or b"<html" in low[:1000]
    )


def validate_file(path: Path):
    if not path.exists():
        return False, "missing"

    try:
        size = path.stat().st_size
    except OSError as e:
        return False, repr(e)

    if size <= 0:
        return False, "zero-byte"

    prefix = read_prefix(path)
    if not prefix:
        return False, "unreadable"

    if looks_like_html(prefix):
        return False, "HTML/SSO login content"

    name = path.name.lower()

    if name.endswith((".tif", ".tiff")):
        if prefix[:4] not in (b"II*\x00", b"MM\x00*"):
            return False, f"invalid TIFF magic {prefix[:16]!r}"

    elif name.endswith((".jpg", ".jpeg")):
        if not prefix.startswith(b"\xff\xd8\xff"):
            return False, f"invalid JPEG magic {prefix[:16]!r}"

    elif name.endswith(".xml"):
        if not prefix.lstrip().startswith(b"<"):
            return False, "invalid XML prefix"
        if looks_like_html(prefix):
            return False, "HTML instead of XML"

    return True, "ok"


def fetch_stac_item(session: requests.Session, product_id: str, attempts=5):
    url = f"{STAC_BASE}/collections/{COLLECTION}/items/{product_id}"
    headers = {
        "Accept": "application/geo+json, application/json;q=0.9",
        "User-Agent": "Stanford-EnMAP-downloader/4.0",
    }

    last = ""
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(
                url,
                params={"f": "json"},
                headers=headers,
                timeout=(30, 120),
            )
            r.raise_for_status()
            item = r.json()
            if not isinstance(item.get("assets"), dict) or not item["assets"]:
                raise RuntimeError("STAC item contains no assets")
            return item
        except Exception as e:
            last = repr(e)
            if attempt < attempts:
                time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"STAC item fetch failed: {last}")


def probe_auth(asset_session: requests.Session, href: str):
    """
    Perform a tiny authenticated ranged GET. This avoids downloading a large
    spectral cube just to discover bad credentials.
    """
    headers = {
        "Range": "bytes=0-31",
        "User-Agent": "Stanford-EnMAP-downloader/4.0",
        "Accept": "*/*",
    }

    try:
        r = asset_session.get(
            href,
            headers=headers,
            stream=True,
            timeout=(30, 90),
            allow_redirects=True,
        )
    except requests.RequestException as e:
        raise AuthenticationFailure(
            f"Authentication probe connection failed: {e!r}"
        )

    final_host = urlparse(r.url).hostname or ""
    ct = (r.headers.get("content-type") or "").lower()

    if r.status_code in (401, 403):
        raise AuthenticationFailure(
            f"DLR returned HTTP {r.status_code}. "
            "Check EnMAP/EOC Geoservice username/password and account access."
        )

    if "sso.eoc.dlr.de" in final_host:
        raise AuthenticationFailure(
            "DLR redirected to the UMS/SSO login page. "
            "The supplied credentials were not accepted for direct download."
        )

    if "text/html" in ct:
        prefix = r.raw.read(512, decode_content=True)
        raise AuthenticationFailure(
            f"DLR returned HTML instead of the asset "
            f"(content-type={ct!r}, final_url={r.url!r}, "
            f"body_prefix={prefix[:120]!r}). "
            "This normally means the download is not authenticated."
        )

    if r.status_code not in (200, 206):
        raise AuthenticationFailure(
            f"Unexpected authentication probe HTTP {r.status_code}: {r.url}"
        )

    # Consume only a tiny prefix then close.
    _ = r.raw.read(32, decode_content=True)
    r.close()


def download_asset(
    asset_session: requests.Session,
    href: str,
    dest: Path,
    mount: Path,
    wait_seconds: int,
    attempts: int = 8,
):
    wait_for_mount(mount, wait_seconds)

    ok, reason = validate_file(dest)
    if ok:
        return "SKIP_EXISTS_VALID", dest.stat().st_size, ""

    # Repair old fake HTML files from v2/v3.
    if dest.exists():
        print(f"    INVALID EXISTING -> removing: {reason}")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")

    if part.exists() and looks_like_html(read_prefix(part)):
        print("    invalid HTML .part -> removing")
        part.unlink()

    last_err = ""

    for attempt in range(1, attempts + 1):
        wait_for_mount(mount, wait_seconds)

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            start = part.stat().st_size if part.exists() else 0
            headers = {
                "User-Agent": "Stanford-EnMAP-downloader/4.0",
                "Accept": "*/*",
            }
            if start:
                headers["Range"] = f"bytes={start}-"

            with asset_session.get(
                href,
                headers=headers,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as r:

                final_host = urlparse(r.url).hostname or ""
                ct = (r.headers.get("content-type") or "").lower()

                if r.status_code in (401, 403):
                    raise AuthenticationFailure(
                        f"HTTP {r.status_code}: credentials/access rejected"
                    )

                if "sso.eoc.dlr.de" in final_host or "text/html" in ct:
                    prefix = r.raw.read(1000, decode_content=True)
                    raise AuthenticationFailure(
                        f"redirected/returned login HTML; "
                        f"final_url={r.url!r}; content_type={ct!r}; "
                        f"body_prefix={prefix[:200]!r}"
                    )

                if r.status_code == 416:
                    print("    HTTP 416 -> resetting this .part")
                    if part.exists():
                        part.unlink()
                    time.sleep(2)
                    continue

                if r.status_code in (408, 425, 429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:500]}"
                    time.sleep(min(60, 2 ** attempt))
                    continue

                r.raise_for_status()

                # Server ignored Range -> safely restart this partial.
                mode = "wb" if (start and r.status_code == 200) else (
                    "ab" if start else "wb"
                )

                with part.open(mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        if not mount_active(mount):
                            raise OSError("SMB mount disappeared during write")
                        f.write(chunk)

            wait_for_mount(mount, wait_seconds)

            if not part.exists() or part.stat().st_size <= 0:
                last_err = "empty/missing .part after download"
                continue

            part.replace(dest)

            ok, reason = validate_file(dest)
            if ok:
                return "DOWNLOADED_VALID", dest.stat().st_size, ""

            last_err = f"post-download validation failed: {reason}"
            if dest.exists():
                dest.unlink()

        except AuthenticationFailure:
            raise

        except KeyboardInterrupt:
            raise

        except (requests.RequestException, OSError) as e:
            last_err = repr(e)

            if not mount_active(mount):
                print(f"    SMB interruption: {dest.name}")
                wait_for_mount(mount, wait_seconds)
                continue

            if attempt < attempts:
                print(f"    retry {attempt}/{attempts}: {e!r}")
                time.sleep(min(60, 2 ** attempt))

    n = part.stat().st_size if part.exists() else 0
    return "FAILED", n, last_err


def main():
    a = parse_args()

    username = (a.username or "").strip()
    if not username:
        username = input("EnMAP / EOC Geoservice username: ").strip()

    password = os.environ.get("ENMAP_PASSWORD")
    if password is None:
        password = getpass.getpass(
            "EnMAP / EOC Geoservice password (hidden): "
        )

    if not username or not password:
        raise SystemExit("Username/password cannot be empty.")

    root = Path(a.root)
    mount = Path(a.mount)
    ensure_dirs(root, mount, a.wait_seconds)

    master = pd.read_csv(a.download_master)
    rows = master[
        (master["sensor"] == "EnMAP")
        & (master["availability_status"] == "RESOLVED_EXACT")
    ].drop_duplicates(subset=["release_ID", "product_id"])

    if a.limit:
        rows = rows.head(a.limit)

    # Public STAC metadata session.
    stac_session = requests.Session()
    stac_session.headers.update({
        "User-Agent": "Stanford-EnMAP-downloader/4.0"
    })

    # Authenticated asset session.
    asset_session = requests.Session()
    asset_session.auth = HTTPBasicAuth(username, password)
    asset_session.headers.update({
        "User-Agent": "Stanford-EnMAP-downloader/4.0"
    })

    log_path = root / "manifests" / "download_log.csv"

    print("=" * 80)
    print("STANFORD ENMAP DOWNLOADER v4 — AUTHENTICATED")
    print("=" * 80)
    print("Resolved products :", len(rows))
    print("Username          :", username)
    print("Output            :", root / "EnMAP")
    print()

    # Get the first live asset and validate authentication before touching 48 scenes.
    if len(rows):
        first_pid = str(rows.iloc[0]["product_id"])
        first_item = fetch_stac_item(stac_session, first_pid)

        first_asset = None
        # Prefer metadata for a cheap auth probe.
        for key in ("metadata", "image"):
            ainfo = first_item.get("assets", {}).get(key)
            if isinstance(ainfo, dict) and isinstance(ainfo.get("href"), str):
                first_asset = ainfo["href"]
                break

        if first_asset is None:
            for ainfo in first_item.get("assets", {}).values():
                if isinstance(ainfo, dict) and isinstance(ainfo.get("href"), str):
                    first_asset = ainfo["href"]
                    break

        if first_asset is None:
            raise SystemExit("No HTTP assets found in first STAC item.")

        print("Checking DLR download authentication...")
        try:
            probe_auth(asset_session, first_asset)
        except AuthenticationFailure as e:
            print()
            print("❌ ENMAP AUTHENTICATION FAILED")
            print(str(e))
            print()
            print("Do not continue until the account can download EnMAP L2A")
            print("from DLR EOC Geoservice.")
            raise SystemExit(3)

        print("✅ DLR authenticated asset access confirmed.")
        print()

    for i, row in enumerate(rows.to_dict("records"), 1):
        rid = str(row["release_ID"])
        pid = str(row["product_id"])

        print(f"[{i}/{len(rows)}] {rid} -> {pid}")

        try:
            item = fetch_stac_item(stac_session, pid)
        except Exception as e:
            print(f"  STAC_ITEM_FAILED: {e}")
            append_log(
                log_path, mount, a.wait_seconds,
                {
                    "sensor": "EnMAP",
                    "release_ID": rid,
                    "product_family": "ENMAP_HSI_L2A",
                    "product_id": pid,
                    "asset": "__STAC_ITEM__",
                    "status": "STAC_ITEM_FAILED",
                    "bytes": 0,
                    "path": "",
                    "error": repr(e),
                },
            )
            continue

        assets = []
        for key, asset in item.get("assets", {}).items():
            href = asset.get("href")
            if isinstance(href, str) and href.startswith(("http://", "https://")):
                assets.append((key, href))

        print(f"  live STAC assets: {len(assets)}")

        wait_for_mount(mount, a.wait_seconds)
        folder = root / "EnMAP" / safe_name(rid) / safe_name(pid)
        folder.mkdir(parents=True, exist_ok=True)

        stac_path = folder / "stac_item_live.json"
        stac_path.write_text(json.dumps(item, indent=2), encoding="utf-8")

        for j, (key, href) in enumerate(assets, 1):
            filename = filename_from_url(href, f"{key}.bin")
            dest = folder / filename

            try:
                status, n, err = download_asset(
                    asset_session,
                    href,
                    dest,
                    mount,
                    a.wait_seconds,
                )
            except AuthenticationFailure as e:
                print(f"  [{j:02d}/{len(assets):02d}] {key:24s} AUTH_FAILED")
                append_log(
                    log_path, mount, a.wait_seconds,
                    {
                        "sensor": "EnMAP",
                        "release_ID": rid,
                        "product_family": "ENMAP_HSI_L2A",
                        "product_id": pid,
                        "asset": key,
                        "status": "AUTH_FAILED",
                        "bytes": 0,
                        "path": str(dest),
                        "error": str(e),
                    },
                )
                print()
                print("❌ Authentication stopped the run.")
                print(str(e))
                raise SystemExit(3)

            print(
                f"  [{j:02d}/{len(assets):02d}] "
                f"{key:24s} "
                f"{status:20s} "
                f"{n/1e6:10.1f} MB"
            )

            append_log(
                log_path, mount, a.wait_seconds,
                {
                    "sensor": "EnMAP",
                    "release_ID": rid,
                    "product_family": "ENMAP_HSI_L2A",
                    "product_id": pid,
                    "asset": key,
                    "status": status,
                    "bytes": n,
                    "path": str(dest),
                    "error": err,
                },
            )

    print()
    print("=" * 80)
    print("ENMAP v4 RUN COMPLETE")
    print("=" * 80)
    print("Run a final content audit before declaring the 48 scenes complete.")
    print("Log:", log_path)


if __name__ == "__main__":
    main()
