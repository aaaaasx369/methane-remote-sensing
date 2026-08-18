#!/usr/bin/env python3
"""
Stanford Landsat downloader v3 — robust to expired SAS tokens AND SMB disconnects.

Key behavior:
  - signs each Planetary Computer asset immediately before download
  - retries with a fresh SAS token on 401/403
  - resumes .part files when Range is supported
  - skips completed non-empty files
  - NEVER creates a fake local /Volumes/engg-leung directory if the SMB share drops
  - stops intentionally when /Volumes/engg-leung is no longer mounted
  - writes a small fallback log to ~/stanford_landsat_fallback_log.csv if the share
    disappears between finishing an asset and writing the main manifest

Safe to rerun after reconnecting the SMB share.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd
import requests

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_ROOT = "/Volumes/engg-leung/dora lin/Stanford_2025_AllSensor"
DEFAULT_MOUNT = "/Volumes/engg-leung"
CHUNK = 8 * 1024 * 1024


class SMBDisconnected(RuntimeError):
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download-master", required=True)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--mount", default=DEFAULT_MOUNT)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9._+\-]+', "_", str(s)).strip("_")


def filename_from_url(url, fallback):
    name = Path(unquote(urlparse(url).path)).name
    return safe_name(name or fallback)


def require_mount(mount_path: Path):
    """
    Critical safety check: on macOS, a disconnected SMB share can make
    /Volumes/engg-leung disappear. We refuse to create anything beneath it
    unless it is a real mounted filesystem.
    """
    if not mount_path.exists() or not os.path.ismount(mount_path):
        raise SMBDisconnected(
            f"SMB mount is not active: {mount_path}\n"
            "Reconnect the share, then rerun the SAME command."
        )


def ensure_storage(root: Path, mount_path: Path):
    require_mount(mount_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "Landsat").mkdir(exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)

    # Verify we can actually write to the server.
    test = root / "manifests" / ".write_test"
    try:
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    except Exception as e:
        raise SMBDisconnected(
            f"SMB is mounted but not writable at {root}: {e!r}"
        )


def fallback_log(row):
    path = Path.home() / "stanford_landsat_fallback_log.csv"
    fields = [
        "timestamp_utc", "sensor", "release_ID", "product_family",
        "product_id", "asset", "status", "bytes", "path", "error"
    ]
    exists = path.exists()
    try:
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            out = {k: row.get(k, "") for k in fields}
            out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            w.writerow(out)
    except Exception:
        pass
    return path


def log_row(log_path: Path, row: dict, mount_path: Path):
    fields = [
        "timestamp_utc", "sensor", "release_ID", "product_family",
        "product_id", "asset", "status", "bytes", "path", "error"
    ]

    try:
        require_mount(mount_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exists = log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            out = {k: row.get(k, "") for k in fields}
            out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            w.writerow(out)
    except Exception as e:
        fb = fallback_log({
            **row,
            "error": f"{row.get('error','')} | MAIN_LOG_FAILED: {e!r}".strip(" |")
        })
        raise SMBDisconnected(
            f"SMB disconnected while writing the manifest.\n"
            f"Fallback log: {fb}"
        )


def request_download(session, signed_url, dest, mount_path, timeout=240):
    require_mount(mount_path)

    if dest.exists() and dest.stat().st_size > 0:
        return "SKIP_EXISTS", dest.stat().st_size, ""

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    try:
        with session.get(
            signed_url,
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as r:
            if r.status_code in (401, 403):
                return f"HTTP_{r.status_code}", start, r.text[:1500]

            if start and r.status_code == 200:
                mode = "wb"
                start = 0
            else:
                mode = "ab" if start else "wb"

            r.raise_for_status()

            with part.open(mode) as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    # Check the SMB mount before every write block.
                    require_mount(mount_path)
                    f.write(chunk)

        require_mount(mount_path)

        if not part.exists() or part.stat().st_size == 0:
            return "FAILED", 0, "zero-byte download"

        part.replace(dest)
        return "DOWNLOADED", dest.stat().st_size, ""

    except SMBDisconnected:
        raise
    except OSError as e:
        # Broken pipe / stale file handle / network filesystem errors.
        if not mount_path.exists() or not os.path.ismount(mount_path):
            raise SMBDisconnected(
                f"SMB disconnected during download of {dest.name}: {e!r}"
            )
        raise


def print_disconnect_banner(exc):
    print()
    print("=" * 80)
    print("SMB SHARE DISCONNECTED — DOWNLOAD STOPPED SAFELY")
    print("=" * 80)
    print(str(exc))
    print()
    print("Already completed non-empty files are preserved.")
    print(".part files are preserved for resume where possible.")
    print()
    print("Reconnect:")
    print("  open 'smb://smb.research-filer.ualberta.ca/engg-leung'")
    print()
    print("Verify:")
    print("  mount | grep engg-leung")
    print("  df -h '/Volumes/engg-leung'")
    print()
    print("Then rerun the EXACT SAME downloader command.")
    print("=" * 80)


def main():
    a = parse_args()

    try:
        import planetary_computer
        from pystac_client import Client
    except Exception as e:
        raise SystemExit(
            "Install dependencies first:\n"
            "python -m pip install -U planetary-computer pystac-client requests pandas\n"
            f"\nOriginal import error: {e!r}"
        )

    root = Path(a.root)
    mount_path = Path(a.mount)

    try:
        ensure_storage(root, mount_path)
    except SMBDisconnected as e:
        print_disconnect_banner(e)
        raise SystemExit(75)

    log_path = root / "manifests" / "download_log.csv"

    master = pd.read_csv(a.download_master)
    rows = master[
        (master["sensor"] == "Landsat") &
        (master["availability_status"] == "RESOLVED_EXACT")
    ].drop_duplicates(subset=["product_id"])

    if a.limit:
        rows = rows.head(a.limit)

    # Keep STAC hrefs unsigned; sign only immediately before each HTTP request.
    client = Client.open(PC_STAC)
    session = requests.Session()

    print("=" * 80)
    print("STANFORD LANDSAT DOWNLOADER v3")
    print("Fresh SAS per asset + SMB disconnect protection")
    print("=" * 80)
    print("Products:", len(rows))
    print("Root:", root)
    print("Mount:", mount_path)
    print()

    try:
        for i, r in enumerate(rows.to_dict("records"), 1):
            require_mount(mount_path)

            rid = str(r["release_ID"])
            pid = str(r["product_id"])
            print(f"[{i}/{len(rows)}] {rid} -> {pid}")

            items = list(
                client.search(
                    collections=["landsat-c2-l2"],
                    ids=[pid],
                    max_items=2,
                ).items()
            )

            if not items:
                print("  ITEM_NOT_FOUND")
                log_row(
                    log_path,
                    {
                        "sensor": "Landsat",
                        "release_ID": rid,
                        "product_family": "Landsat_C2_L2",
                        "product_id": pid,
                        "status": "ITEM_NOT_FOUND",
                    },
                    mount_path,
                )
                continue

            item = items[0]
            folder = root / "Landsat" / safe_name(rid) / safe_name(pid)
            require_mount(mount_path)
            folder.mkdir(parents=True, exist_ok=True)

            meta = folder / "stac_item_unsigned.json"
            if not meta.exists():
                meta.write_text(
                    json.dumps(item.to_dict(), indent=2),
                    encoding="utf-8",
                )

            for key, asset in item.assets.items():
                require_mount(mount_path)

                raw_href = asset.href
                if not raw_href or not raw_href.startswith("http"):
                    continue

                dest = folder / (
                    f"{safe_name(key)}__"
                    f"{filename_from_url(raw_href, key + '.bin')}"
                )

                if dest.exists() and dest.stat().st_size > 0:
                    status, n, err = "SKIP_EXISTS", dest.stat().st_size, ""
                else:
                    status, n, err = "FAILED", 0, ""

                    for attempt in range(1, 5):
                        require_mount(mount_path)
                        try:
                            signed = planetary_computer.sign(raw_href)
                            status, n, err = request_download(
                                session,
                                signed,
                                dest,
                                mount_path,
                            )

                            if status in ("DOWNLOADED", "SKIP_EXISTS"):
                                break

                            # 401/403 commonly means stale/expired SAS.
                            if status in ("HTTP_401", "HTTP_403"):
                                time.sleep(min(10, 2 * attempt))
                                continue

                            time.sleep(min(10, 2 * attempt))

                        except SMBDisconnected:
                            raise
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            status = "FAILED"
                            err = repr(e)
                            time.sleep(min(10, 2 * attempt))

                print(
                    f"  {key:24s} "
                    f"{status:14s} "
                    f"{n / 1e6:10.1f} MB"
                )

                log_row(
                    log_path,
                    {
                        "sensor": "Landsat",
                        "release_ID": rid,
                        "product_family": "Landsat_C2_L2",
                        "product_id": pid,
                        "asset": key,
                        "status": status,
                        "bytes": n,
                        "path": str(dest),
                        "error": err,
                    },
                    mount_path,
                )

    except SMBDisconnected as e:
        print_disconnect_banner(e)
        raise SystemExit(75)

    print()
    print("=" * 80)
    print("LANDSAT RUN COMPLETE")
    print("=" * 80)
    print("Log:", log_path)
    print("Safe to rerun: completed files are skipped.")


if __name__ == "__main__":
    main()
