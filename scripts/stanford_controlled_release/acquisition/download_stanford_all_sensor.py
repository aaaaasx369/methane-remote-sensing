#!/usr/bin/env python3
"""
Download all resolved Stanford 2025 public imagery to the UAlberta lab SMB.

Sensors handled:
  - Sentinel-2 L2A: full CDSE product ZIP ($value), 174 resolved scenes
  - Landsat 8/9 C2 L2: all STAC assets from Microsoft Planetary Computer, 85 scenes
  - EnMAP HSI L2A: all STAC assets from DLR EOC, 48 scenes
  - EMIT L2A RFL + L2B CH4 enhancement: NASA Earthdata via earthaccess, 18 product rows / 9 events

Inputs:
  --download-master  01_resolved_products_download_master.csv
  --s2-matches       02_selected_scene_matches.csv

Default output:
  /Volumes/engg-leung/dora lin/Stanford_2025_AllSensor

The downloader is resumable:
  - existing non-empty files are skipped
  - interrupted HTTP downloads use .part files and Range requests when possible
  - progress/status is appended to manifests/download_log.csv

Credentials:
  Sentinel-2:
    Uses CDSE_USERNAME / CDSE_PASSWORD environment variables if present;
    otherwise prompts interactively. CDSE_TOTP is optional for 2FA.
  EMIT:
    Uses earthaccess.login(strategy="all", persist=True), so it can use
    environment variables, ~/.netrc, or an interactive Earthdata Login prompt.

Install:
  python -m pip install requests pandas pystac-client planetary-computer earthaccess
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd
import requests

DEFAULT_ROOT = "/Volumes/engg-leung/dora lin/Stanford_2025_AllSensor"

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
DLR_ITEM = "https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items/{item_id}"
CDSE_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

CHUNK = 8 * 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download-master", required=True,
                   help="01_resolved_products_download_master.csv")
    p.add_argument("--s2-matches", required=True,
                   help="02_selected_scene_matches.csv from stanford_s2_scene_match")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--sensors", default="all",
                   help="all or comma list: sentinel2,landsat,enmap,emit")
    p.add_argument("--threads", type=int, default=4,
                   help="earthaccess threads for EMIT")
    p.add_argument("--limit", type=int, default=0,
                   help="Debug limit per sensor; 0 = no limit")
    return p.parse_args()


def ensure_root(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    test = root / ".write_test"
    test.write_text("ok", encoding="utf-8")
    test.unlink()
    for sub in ["Sentinel2", "Landsat", "EnMAP", "EMIT", "manifests", "logs"]:
        (root / sub).mkdir(exist_ok=True)


def safe_name(s):
    s = str(s)
    return re.sub(r'[^A-Za-z0-9._+\-]+', "_", s).strip("_")


def filename_from_url(url, fallback="download.bin"):
    path = unquote(urlparse(url).path)
    name = Path(path).name
    return safe_name(name or fallback)


def disk_free_text(path: Path):
    u = shutil.disk_usage(path)
    return f"{u.free / (1024**4):.2f} TiB free"


def log_row(log_path: Path, row: dict):
    exists = log_path.exists()
    fields = [
        "timestamp_utc", "sensor", "release_ID", "product_family",
        "product_id", "asset", "status", "bytes", "path", "error"
    ]
    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        r = {k: row.get(k, "") for k in fields}
        r["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        w.writerow(r)


def stream_download(session: requests.Session, url: str, dest: Path,
                    headers=None, timeout=180, retries=5):
    """
    Resumable HTTP download. Returns (status, bytes_on_disk, error).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "SKIP_EXISTS", dest.stat().st_size, ""

    part = dest.with_suffix(dest.suffix + ".part")
    last_err = ""

    for attempt in range(1, retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            req_headers = dict(headers or {})
            if start > 0:
                req_headers["Range"] = f"bytes={start}-"

            with session.get(
                url,
                headers=req_headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True
            ) as r:
                # Auth failures should be handled by caller.
                if r.status_code in (401, 403):
                    return f"HTTP_{r.status_code}", start, r.text[:500]

                # If server ignores Range, start over.
                mode = "ab" if (start > 0 and r.status_code == 206) else "wb"
                if start > 0 and r.status_code == 200:
                    start = 0

                r.raise_for_status()

                with part.open(mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)

            if part.stat().st_size <= 0:
                raise RuntimeError("zero-byte download")

            part.replace(dest)
            return "DOWNLOADED", dest.stat().st_size, ""

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = repr(e)
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))

    n = part.stat().st_size if part.exists() else 0
    return "FAILED", n, last_err


# ----------------------------------------------------------------------
# Sentinel-2 CDSE
# ----------------------------------------------------------------------

class CDSEAuth:
    def __init__(self):
        self.username = os.getenv("CDSE_USERNAME") or input("CDSE username: ").strip()
        self.password = os.getenv("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")
        self.totp = os.getenv("CDSE_TOTP", "").strip()
        self.access_token = None
        self.refresh_token = None
        self.expires_at = datetime.now(timezone.utc)

    def _request_password_token(self):
        data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }
        if self.totp:
            data["totp"] = self.totp
        r = requests.post(CDSE_TOKEN, data=data, timeout=60)
        r.raise_for_status()
        js = r.json()
        self.access_token = js["access_token"]
        self.refresh_token = js.get("refresh_token")
        self.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(60, int(js.get("expires_in", 600)) - 60)
        )

    def _refresh(self):
        if not self.refresh_token:
            self._request_password_token()
            return
        data = {
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        r = requests.post(CDSE_TOKEN, data=data, timeout=60)
        if r.status_code >= 400:
            self._request_password_token()
            return
        js = r.json()
        self.access_token = js["access_token"]
        self.refresh_token = js.get("refresh_token", self.refresh_token)
        self.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(60, int(js.get("expires_in", 600)) - 60)
        )

    def headers(self, force=False):
        if force or not self.access_token:
            self._request_password_token()
        elif datetime.now(timezone.utc) >= self.expires_at:
            self._refresh()
        return {"Authorization": f"Bearer {self.access_token}"}


def download_sentinel2(df, root, log_path, limit=0):
    df = df[df["match_status"].eq("PASS") | df["match_status"].eq("REVIEW_TIME_DELTA")].copy()
    # The prior audit established all these rows as resolved; dedupe on product UUID.
    df = df.drop_duplicates(subset=["cdse_product_id"])
    if limit:
        df = df.head(limit)

    print(f"\nSentinel-2 products to process: {len(df)}")
    auth = CDSEAuth()
    session = requests.Session()

    for i, r in enumerate(df.to_dict("records"), 1):
        rid = str(r["release_ID"])
        pid = str(r["cdse_product_id"])
        pname = str(r["cdse_product_name"])
        folder = root / "Sentinel2" / safe_name(rid)
        dest = folder / f"{safe_name(pname)}.zip"

        print(f"[S2 {i}/{len(df)}] {rid} -> {pname}")
        if dest.exists() and dest.stat().st_size > 0:
            status, n, err = "SKIP_EXISTS", dest.stat().st_size, ""
        else:
            url = CDSE_DOWNLOAD.format(product_id=pid)
            status, n, err = stream_download(
                session, url, dest, headers=auth.headers()
            )
            if status in ("HTTP_401", "HTTP_403"):
                status, n, err = stream_download(
                    session, url, dest, headers=auth.headers(force=True)
                )

        log_row(log_path, {
            "sensor": "Sentinel-2", "release_ID": rid,
            "product_family": "S2_MSI_L2A_FULL_PRODUCT",
            "product_id": pid, "asset": "full_product_zip",
            "status": status, "bytes": n, "path": str(dest), "error": err
        })
        print(" ", status, f"{n/1e6:.1f} MB")


# ----------------------------------------------------------------------
# Landsat Planetary Computer
# ----------------------------------------------------------------------

def get_pc_client():
    try:
        import planetary_computer
        from pystac_client import Client
    except Exception as e:
        raise RuntimeError(
            "Missing Landsat dependencies. Run: "
            "python -m pip install pystac-client planetary-computer"
        ) from e
    return Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)


def download_landsat(master, root, log_path, limit=0):
    rows = master[
        (master["sensor"] == "Landsat") &
        (master["recommended_action"] == "DOWNLOAD")
    ].copy()
    rows = rows.drop_duplicates(subset=["product_id"])
    if limit:
        rows = rows.head(limit)

    print(f"\nLandsat products to process: {len(rows)}")
    client = get_pc_client()
    session = requests.Session()

    for i, r in enumerate(rows.to_dict("records"), 1):
        rid = str(r["release_ID"])
        pid = str(r["product_id"])
        folder = root / "Landsat" / safe_name(rid) / safe_name(pid)
        folder.mkdir(parents=True, exist_ok=True)

        print(f"[Landsat {i}/{len(rows)}] {rid} -> {pid}")

        search = client.search(
            collections=["landsat-c2-l2"],
            ids=[pid],
            max_items=2
        )
        items = list(search.items())
        if not items:
            log_row(log_path, {
                "sensor": "Landsat", "release_ID": rid,
                "product_family": "Landsat_C2_L2",
                "product_id": pid, "status": "ITEM_NOT_FOUND",
                "error": "Planetary Computer STAC item not found"
            })
            print("  ITEM_NOT_FOUND")
            continue

        item = items[0]
        meta_path = folder / "stac_item.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(item.to_dict(), indent=2), encoding="utf-8")

        for asset_key, asset in item.assets.items():
            href = asset.href
            if not href or not href.startswith("http"):
                continue
            fname = filename_from_url(href, fallback=f"{asset_key}.bin")
            dest = folder / f"{safe_name(asset_key)}__{fname}"
            status, n, err = stream_download(session, href, dest)

            log_row(log_path, {
                "sensor": "Landsat", "release_ID": rid,
                "product_family": "Landsat_C2_L2",
                "product_id": pid, "asset": asset_key,
                "status": status, "bytes": n, "path": str(dest), "error": err
            })
            print(f"  {asset_key}: {status} {n/1e6:.1f} MB")


# ----------------------------------------------------------------------
# EnMAP DLR STAC
# ----------------------------------------------------------------------

def download_enmap(master, root, log_path, limit=0):
    rows = master[
        (master["sensor"] == "EnMAP") &
        (master["recommended_action"] == "DOWNLOAD")
    ].copy()
    rows = rows.drop_duplicates(subset=["product_id"])
    if limit:
        rows = rows.head(limit)

    print(f"\nEnMAP products to process: {len(rows)}")
    session = requests.Session()
    session.headers.update({"User-Agent": "Stanford-EnMAP-Downloader/1.0"})

    for i, r in enumerate(rows.to_dict("records"), 1):
        rid = str(r["release_ID"])
        pid = str(r["product_id"])
        folder = root / "EnMAP" / safe_name(rid) / safe_name(pid)
        folder.mkdir(parents=True, exist_ok=True)

        print(f"[EnMAP {i}/{len(rows)}] {rid} -> {pid}")

        item_url = DLR_ITEM.format(item_id=pid)
        resp = session.get(
            item_url,
            params={"f": "application/geo+json"},
            timeout=90
        )
        if resp.status_code != 200:
            log_row(log_path, {
                "sensor": "EnMAP", "release_ID": rid,
                "product_family": "ENMAP_HSI_L2A",
                "product_id": pid, "status": f"ITEM_HTTP_{resp.status_code}",
                "error": resp.text[:500]
            })
            print(" ", f"ITEM_HTTP_{resp.status_code}")
            continue

        item = resp.json()
        (folder / "stac_item.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )

        for asset_key, asset in (item.get("assets") or {}).items():
            href = (asset or {}).get("href", "")
            if not href or not href.startswith("http"):
                continue
            fname = filename_from_url(href, fallback=f"{asset_key}.bin")
            dest = folder / f"{safe_name(asset_key)}__{fname}"
            status, n, err = stream_download(session, href, dest)

            log_row(log_path, {
                "sensor": "EnMAP", "release_ID": rid,
                "product_family": "ENMAP_HSI_L2A",
                "product_id": pid, "asset": asset_key,
                "status": status, "bytes": n, "path": str(dest), "error": err
            })
            print(f"  {asset_key}: {status} {n/1e6:.1f} MB")


# ----------------------------------------------------------------------
# EMIT NASA Earthdata
# ----------------------------------------------------------------------

def download_emit(master, root, log_path, threads=4, limit=0):
    try:
        import earthaccess
    except Exception as e:
        raise RuntimeError(
            "Missing earthaccess. Run: python -m pip install earthaccess"
        ) from e

    rows = master[
        (master["sensor"] == "EMIT") &
        (master["recommended_action"] == "DOWNLOAD")
    ].copy()
    rows = rows.drop_duplicates(subset=["product_family", "product_id"])
    if limit:
        rows = rows.head(limit)

    print(f"\nEMIT products to process: {len(rows)}")
    print("Earthdata login:")
    earthaccess.login(strategy="all", persist=True)

    family_map = {
        "EMIT_L2A_RFL": ("EMITL2ARFL", "001"),
        "EMIT_L2B_CH4ENH": ("EMITL2BCH4ENH", "002"),
    }

    for i, r in enumerate(rows.to_dict("records"), 1):
        rid = str(r["release_ID"])
        family = str(r["product_family"])
        pid = str(r["product_id"])
        short_name, version = family_map[family]
        folder = root / "EMIT" / safe_name(rid) / safe_name(family)
        folder.mkdir(parents=True, exist_ok=True)

        print(f"[EMIT {i}/{len(rows)}] {rid} -> {family} -> {pid}")

        try:
            granules = earthaccess.search_data(
                short_name=short_name,
                version=version,
                granule_name=f"{pid}*",
                downloadable=True,
                count=-1,
            )
            if not granules:
                log_row(log_path, {
                    "sensor": "EMIT", "release_ID": rid,
                    "product_family": family, "product_id": pid,
                    "status": "GRANULE_NOT_FOUND",
                    "error": "earthaccess search returned no granules"
                })
                print("  GRANULE_NOT_FOUND")
                continue

            paths = earthaccess.download(
                granules,
                local_path=folder,
                threads=threads,
                show_progress=True,
                force=False,
            )
            total = 0
            for p in paths:
                try:
                    total += Path(p).stat().st_size
                except Exception:
                    pass
            log_row(log_path, {
                "sensor": "EMIT", "release_ID": rid,
                "product_family": family, "product_id": pid,
                "asset": "earthaccess_granule",
                "status": "DOWNLOADED_OR_EXISTS",
                "bytes": total,
                "path": " | ".join(str(p) for p in paths)
            })
            print(" ", f"DOWNLOADED_OR_EXISTS {total/1e6:.1f} MB")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log_row(log_path, {
                "sensor": "EMIT", "release_ID": rid,
                "product_family": family, "product_id": pid,
                "status": "FAILED", "error": repr(e)
            })
            print(" ", "FAILED", repr(e))


def main():
    a = parse_args()
    root = Path(a.root)
    ensure_root(root)

    print("=" * 80)
    print("STANFORD 2025 ALL-SENSOR DOWNLOADER")
    print("=" * 80)
    print("Output:", root)
    print("Storage:", disk_free_text(root))
    print()

    master = pd.read_csv(a.download_master)
    s2 = pd.read_csv(a.s2_matches)

    sensors = {x.strip().lower() for x in a.sensors.split(",")}
    if "all" in sensors:
        sensors = {"sentinel2", "landsat", "enmap", "emit"}

    log_path = root / "manifests" / "download_log.csv"

    # Save exact input manifests alongside downloaded data.
    shutil.copy2(a.download_master, root / "manifests" / "resolved_products_download_master.csv")
    shutil.copy2(a.s2_matches, root / "manifests" / "sentinel2_selected_scene_matches.csv")

    if "landsat" in sensors:
        download_landsat(master, root, log_path, a.limit)

    if "enmap" in sensors:
        download_enmap(master, root, log_path, a.limit)

    if "emit" in sensors:
        download_emit(master, root, log_path, a.threads, a.limit)

    if "sentinel2" in sensors:
        download_sentinel2(s2, root, log_path, a.limit)

    print()
    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print("Output:", root)
    print("Log   :", log_path)
    print("Storage:", disk_free_text(root))
    print()
    print("You can rerun the same command after interruption; existing files are skipped.")


if __name__ == "__main__":
    main()
