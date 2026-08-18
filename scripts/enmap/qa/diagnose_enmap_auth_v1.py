#!/usr/bin/env python3
"""
diagnose_enmap_auth_v1.py

Read the first current EnMAP asset URL from the V3 smoke-test plan and test
DLR EOC Geoservice Basic authentication using both likely EnMAP usernames:

  1) doraaa
  2) doraaa-cat1distributor

The password is prompted securely and is never written to disk.

This follows the DLR UKIS EnMAP tutorial pattern:
  urllib3.make_headers(basic_auth="USER:PASSWORD")
  http.request("GET", asset.href, headers=header, preload_content=False)

No complete EnMAP file is downloaded; only a small initial response is read.
"""

from __future__ import annotations

import getpass
from pathlib import Path

import pandas as pd
import urllib3


PLAN = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary72h_download_v3"
    / "download_plan_limit.csv"
)

USERNAMES = [
    "doraaa",
    "doraaa-cat1distributor",
]


def main():
    if not PLAN.exists():
        raise SystemExit(f"Plan not found: {PLAN}")

    df = pd.read_csv(PLAN, low_memory=False)
    if df.empty:
        raise SystemExit("Download plan is empty.")

    # Prefer tiny metadata asset for diagnosis.
    if "asset_key" in df.columns:
        m = df["asset_key"].astype(str).str.lower().eq("metadata")
        row = df[m].iloc[0] if m.any() else df.iloc[0]
    else:
        row = df.iloc[0]

    url = str(row["url"])

    print("=" * 80)
    print("ENMAP AUTH DIAGNOSTIC")
    print("=" * 80)
    print("Asset:")
    print(row.get("resolved_scene_id", row.get("original_scene_id", "")))
    print()
    print("URL:")
    print(url)
    print()
    print("This will test authentication without saving the password.")
    password = getpass.getpass("EnMAP / IPS password: ")

    if not password:
        raise SystemExit("Empty password; aborting.")

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=30.0, read=45.0),
        retries=False,
    )

    for username in USERNAMES:
        print()
        print("-" * 80)
        print(f"Testing username: {username}")

        headers = urllib3.make_headers(
            basic_auth=f"{username}:{password}"
        )
        headers["User-Agent"] = "UAlberta-EnMAP-Auth-Diagnostic/1.0"
        headers["Range"] = "bytes=0-1023"

        try:
            r = http.request(
                "GET",
                url,
                headers=headers,
                preload_content=False,
                redirect=False,
            )

            print(f"HTTP status : {r.status}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Location    : {r.headers.get('Location')}")
            print(f"Content-Range: {r.headers.get('Content-Range')}")

            # Read only a small amount, then close.
            data = r.read(1024)
            print(f"Bytes read  : {len(data)}")

            if r.status in (200, 206):
                print("RESULT      : SUCCESS")
            elif r.status in (301, 302, 303, 307, 308):
                print("RESULT      : REDIRECTED TO LOGIN / SSO")
            elif r.status in (401, 403):
                print("RESULT      : AUTHENTICATION / AUTHORIZATION FAILED")
            elif r.status == 404:
                print("RESULT      : 404 — AUTH OR ASSET-PATH ISSUE")
            else:
                print("RESULT      : OTHER HTTP STATUS")

            r.release_conn()

        except Exception as e:
            print(f"ERROR       : {type(e).__name__}: {e}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Paste the two username result blocks back to ChatGPT.")


if __name__ == "__main__":
    main()
