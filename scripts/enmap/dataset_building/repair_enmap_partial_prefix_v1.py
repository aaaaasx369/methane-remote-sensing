#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import urllib3

DEFAULT_PLAN = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary721_download_v2"
    / "download_plan_full.csv"
)

DEFAULT_ROOT = Path(
    "/Volumes/engg-leung/dora lin/EnMAP_MethaneFuse/"
    "01_raw_L2A/primary72h_nominal"
)

DEFAULT_SCENE = (
    "ENMAP01-____L2A-DT0000147154_20250813T080840Z_003_"
    "V010502_20250814T082715Z"
)

USERNAME = "doraaa"


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
        (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    )


def is_tiff_magic(b: bytes) -> bool:
    return b[:4] in (
        b"II*\x00",
        b"MM\x00*",
        b"II+\x00",
        b"MM\x00+",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--prefix-bytes", type=int, default=1075)
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()

    scene = args.scene
    n = args.prefix_bytes
    plan_path = Path(args.plan).expanduser().resolve()
    root = Path(args.root).expanduser()

    if not plan_path.exists():
        raise SystemExit(f"Plan not found: {plan_path}")

    plan = pd.read_csv(plan_path, low_memory=False)
    m = (
        plan["current_scene_id"].astype(str).eq(scene)
        & plan["asset_key"].astype(str).str.lower().eq("image")
    )
    if not m.any():
        raise SystemExit(f"Could not find image asset for scene in plan: {scene}")

    row = plan[m].iloc[0]
    original_url = str(row["original_url"])
    repaired_url = (
        str(row["repaired_url"])
        if "repaired_url" in row and pd.notna(row["repaired_url"])
        else insert_scene_directory(original_url, scene)
    )

    filename = Path(str(row["destination"])).name
    part = root / scene / f"{filename}.part"

    if not part.exists():
        raise SystemExit(f".part not found: {part}")

    size_before = part.stat().st_size
    with open(part, "rb") as f:
        old_prefix = f.read(16)

    print("=" * 88)
    print("ENMAP PARTIAL PREFIX REPAIR")
    print("=" * 88)
    print("Scene       :", scene)
    print("Part        :", part)
    print("Part size   :", size_before, "bytes")
    print("Replace     :", n, "leading bytes")
    print("Old prefix  :", old_prefix)

    if is_tiff_magic(old_prefix):
        print("\nAlready has a TIFF header. Nothing to repair.")
        return

    password = getpass.getpass(f"IPS password for {USERNAME}: ")
    if not password:
        raise SystemExit("Empty password; aborting.")

    headers = urllib3.make_headers(basic_auth=f"{USERNAME}:{password}")
    headers["User-Agent"] = "UAlberta-EnMAP-Prefix-Repair/1.0"
    headers["Range"] = f"bytes=0-{n-1}"

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=45.0, read=120.0),
        retries=False,
    )

    print("\nFetching correct prefix from REPAIRED asset URL...")

    r = http.request(
        "GET",
        repaired_url,
        headers=headers,
        preload_content=False,
        redirect=False,
    )

    status = int(r.status)
    content_range = r.headers.get("Content-Range")
    data = r.read(n)
    r.release_conn()

    print("HTTP         :", status)
    print("Content-Range:", content_range)
    print("Bytes read   :", len(data))
    print("New prefix   :", data[:16])

    if status not in (200, 206):
        raise SystemExit(f"Unexpected HTTP status: {status}")
    if len(data) != n:
        raise SystemExit(f"Expected exactly {n} bytes, got {len(data)}")
    if not is_tiff_magic(data):
        raise SystemExit("Fetched prefix is not TIFF data. Refusing to modify .part.")

    with open(part, "r+b") as f:
        f.seek(0)
        f.write(data)
        f.flush()

    with open(part, "rb") as f:
        verify = f.read(16)

    if not is_tiff_magic(verify):
        raise SystemExit("Repair verification failed.")

    print("\n✅ PREFIX REPAIRED")
    print("Part size unchanged:", part.stat().st_size, "bytes")
    print("TIFF prefix        :", verify)
    print("\nDo NOT rename the file yet; it is still a partial download.")
    print("Resume it with the V4 downloader.")


if __name__ == "__main__":
    main()
