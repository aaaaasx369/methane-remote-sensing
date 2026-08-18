#!/usr/bin/env python3
"""
diagnose_enmap_404_path_repair_v1.py

For the 5 exact-current EnMAP scenes already audited, test whether 404 asset
URLs become valid when the product scene ID directory is inserted before the
filename.

Observed pattern:
  failing:
    .../DT.../01/<SCENE>-METADATA.XML

  working:
    .../DT.../01/<SCENE>/<SCENE>-METADATA.XML

This script:
- uses IPS username "doraaa"
- prompts securely for the password
- reads only bytes 0-1023
- never downloads a full product
- writes a small CSV audit

Input:
  ~/methane_release_project/enmap_exact_resolution_audit_v1/
      exact_resolution_limit5.csv

Output:
  ~/methane_release_project/enmap_exact_resolution_audit_v1/
      exact_metadata_path_repair_test.csv
"""

from __future__ import annotations

import getpass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import urllib3


INPUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_exact_resolution_audit_v1"
    / "exact_resolution_limit5.csv"
)
OUTPUT = INPUT.parent / "exact_metadata_path_repair_test.csv"
USERNAME = "doraaa"


def insert_scene_directory(url: str, scene_id: str) -> str:
    parts = urlsplit(url)
    path = parts.path
    pieces = path.rstrip("/").split("/")

    if not pieces:
        return url

    filename = pieces[-1]

    # Already has scene folder immediately before filename.
    if len(pieces) >= 2 and pieces[-2] == scene_id:
        return url

    new_path = "/".join(pieces[:-1] + [scene_id, filename])
    if path.startswith("/") and not new_path.startswith("/"):
        new_path = "/" + new_path

    return urlunsplit(
        (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    )


def probe(http, url: str, password: str):
    headers = urllib3.make_headers(
        basic_auth=f"{USERNAME}:{password}"
    )
    headers["User-Agent"] = "UAlberta-EnMAP-Path-Repair-Test/1.0"
    headers["Range"] = "bytes=0-1023"

    try:
        r = http.request(
            "GET",
            url,
            headers=headers,
            preload_content=False,
            redirect=False,
        )

        status = int(r.status)
        ctype = r.headers.get("Content-Type")
        crange = r.headers.get("Content-Range")
        location = r.headers.get("Location")
        data = r.read(1024)
        nbytes = len(data)
        r.release_conn()

        return {
            "http_status": status,
            "content_type": ctype,
            "content_range": crange,
            "location": location,
            "bytes_read": nbytes,
        }
    except Exception as e:
        return {
            "http_status": None,
            "content_type": None,
            "content_range": None,
            "location": None,
            "bytes_read": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    password = getpass.getpass(
        f"IPS password for {USERNAME}: "
    )
    if not password:
        raise SystemExit("Empty password; aborting.")

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=30.0, read=60.0),
        retries=False,
    )

    out_rows = []

    print("=" * 92)
    print("ENMAP EXACT-CURRENT METADATA PATH REPAIR TEST")
    print("=" * 92)

    for i, r in df.iterrows():
        scene = str(r["current_scene_id"])
        original_url = str(r["metadata_href"])
        repaired_url = insert_scene_directory(
            original_url, scene
        )

        print()
        print(f"[{i+1}/{len(df)}]")
        print("Scene:", scene)

        original = probe(http, original_url, password)
        print("ORIGINAL")
        print("  HTTP :", original.get("http_status"))
        print("  Type :", original.get("content_type"))
        print("  Range:", original.get("content_range"))

        if repaired_url == original_url:
            repaired = original.copy()
            print("REPAIRED")
            print("  URL already contains scene directory")
            print("  HTTP :", repaired.get("http_status"))
        else:
            print("REPAIRED URL:")
            print(repaired_url)
            repaired = probe(http, repaired_url, password)
            print("REPAIRED")
            print("  HTTP :", repaired.get("http_status"))
            print("  Type :", repaired.get("content_type"))
            print("  Range:", repaired.get("content_range"))

        if original.get("http_status") in (200, 206):
            result = "ORIGINAL_WORKS"
        elif (
            original.get("http_status") == 404
            and repaired.get("http_status") in (200, 206)
        ):
            result = "REPAIRED_PATH_WORKS"
        elif repaired.get("http_status") == 404:
            result = "STILL_404"
        elif repaired.get("http_status") in (401, 403):
            result = "AUTH_FAILURE"
        else:
            result = "OTHER"

        print("RESULT:", result)

        out_rows.append(
            {
                "scene_id": scene,
                "original_url": original_url,
                "original_http": original.get("http_status"),
                "repaired_url": repaired_url,
                "repaired_http": repaired.get("http_status"),
                "repaired_content_type": repaired.get("content_type"),
                "repaired_content_range": repaired.get("content_range"),
                "result": result,
            }
        )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT, index=False)

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(out["result"].value_counts(dropna=False).to_string())
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
