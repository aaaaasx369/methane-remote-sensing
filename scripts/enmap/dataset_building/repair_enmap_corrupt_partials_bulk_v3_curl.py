#!/usr/bin/env python3
"""
repair_enmap_corrupt_partials_bulk_v3_curl.py

Bulk repair EnMAP TIFF .part files contaminated by a leading HTML error body.

Why V3:
The EnMAP image URLs have already been proven downloadable by the working
curl-based bulk downloader. urllib3 prefix probes can still fail on these
endpoints. V3 therefore obtains only the first N remote bytes using curl with
the SAME Basic-auth transport style as the downloader.

It starts a normal authenticated GET, reads exactly N bytes from curl stdout,
then terminates curl immediately. No full image is downloaded.

Default is DRY-RUN. Use --apply only after a successful dry-run.

Input:
  ~/methane_release_project/enmap_primary721_download_v2/
      download_checkpoint_full.csv

Output:
  ~/methane_release_project/enmap_corrupt_prefix_repair_v3/
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DEFAULT_CHECKPOINT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary721_download_v2"
    / "download_checkpoint_full.csv"
)

DEFAULT_STATE = (
    Path.home()
    / "methane_release_project"
    / "enmap_corrupt_prefix_repair_v3"
)

USERNAME = "doraaa"


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

    # Consume only immediate CR/LF after HTML.
    for _ in range(4):
        if n < len(head) and head[n:n+1] in (b"\r", b"\n"):
            n += 1
        else:
            break

    return n


def curl_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def fetch_prefix_with_curl(
    url: str,
    username: str,
    password: str,
    n: int,
    timeout: int = 45,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Start a normal authenticated curl GET, capture exactly n body bytes,
    then terminate curl. This intentionally does NOT use an HTTP Range header.
    """
    userpass = curl_escape(f"{username}:{password}")
    url_escaped = curl_escape(url)

    config = "\n".join([
        f'user = "{userpass}"',
        "basic",
        "location",
        "fail",
        "silent",
        "show-error",
        "connect-timeout = 30",
        f"max-time = {timeout}",
        f'url = "{url_escaped}"',
        "",
    ]).encode()

    proc = subprocess.Popen(
        ["curl", "--config", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    proc.stdin.write(config)
    proc.stdin.close()

    try:
        data = proc.stdout.read(n)
    except Exception as e:
        proc.kill()
        proc.wait()
        return None, f"stdout read failed: {type(e).__name__}: {e}"

    # We have what we need; stop the large remote transfer immediately.
    if proc.poll() is None:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    err = proc.stderr.read().decode("utf-8", errors="replace").strip()

    if len(data) != n:
        return None, (
            f"SHORT_READ:{len(data)}/{n}; "
            f"curl_rc={proc.returncode}; stderr={err[:300]}"
        )

    if not is_tiff_magic(data):
        return None, (
            f"NOT_TIFF:{data[:16]!r}; "
            f"curl_rc={proc.returncode}; stderr={err[:300]}"
        )

    return data, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE))
    ap.add_argument("--limit", type=int, default=0, help="0 = all corrupt rows")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state = Path(args.state_dir).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)

    backup_dir = state / "prefix_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    df = pd.read_csv(checkpoint, low_memory=False)

    corrupt = df[
        df["download_status"]
        .fillna("")
        .astype(str)
        .str.startswith("CORRUPT_PART_PREFIX:")
    ].copy()

    if args.limit > 0:
        corrupt = corrupt.head(args.limit).copy()

    print("=" * 92)
    print("ENMAP BULK CORRUPT-PREFIX REPAIR V3 — CURL STREAM")
    print("=" * 92)
    print("Checkpoint rows :", len(df))
    print("Corrupt rows    :", len(corrupt))
    print("Mode            :", "APPLY" if args.apply else "DRY-RUN")
    print()

    if corrupt.empty:
        raise SystemExit("No CORRUPT_PART_PREFIX rows found.")

    password = getpass.getpass(f"IPS password for {USERNAME}: ")
    if not password:
        raise SystemExit("Empty password; aborting.")

    results = []

    for pos, (_, r) in enumerate(corrupt.iterrows(), start=1):
        final_path = Path(str(r["destination"]))
        part = Path(str(final_path) + ".part")

        scene = str(r.get("current_scene_id", ""))
        asset = str(r.get("asset_key", ""))

        base = {
            "current_scene_id": scene,
            "asset_key": asset,
            "part_path": str(part),
            "part_size_bytes": part.stat().st_size if part.exists() else 0,
        }

        print(f"[{pos}/{len(corrupt)}] {scene} :: {asset}")

        if not part.exists():
            print("  MISSING_PART")
            results.append({**base, "status": "MISSING_PART"})
            continue

        if final_path.suffix.lower() not in (".tif", ".tiff"):
            print("  NOT_TIFF_PART")
            results.append({**base, "status": "NOT_TIFF_PART"})
            continue

        with open(part, "rb") as f:
            first16 = f.read(16)

        if is_tiff_magic(first16):
            print("  ALREADY_TIFF")
            results.append({
                **base,
                "status": "ALREADY_TIFF",
                "prefix_len": 0,
            })
            continue

        n = infer_html_prefix_len(part)
        if not n:
            print("  HTML_END_NOT_FOUND")
            results.append({
                **base,
                "status": "HTML_END_NOT_FOUND",
                "first16": repr(first16),
            })
            continue

        original_url = str(r.get("original_url", ""))
        repaired_url = str(r.get("repaired_url", ""))
        path_style = str(r.get("availability_path_style", ""))

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
            # Most contaminated rows originated from a bad original URL, then
            # real TIFF data arrived from the repaired scene-directory URL.
            candidates = [
                ("REPAIRED", repaired_url),
                ("ORIGINAL", original_url),
            ]

        good = None
        errors = []

        for style, url in candidates:
            if not url or url == "nan":
                continue

            data, err = fetch_prefix_with_curl(
                url=url,
                username=USERNAME,
                password=password,
                n=n,
            )

            if data is not None:
                good = (style, url, data)
                break

            errors.append(f"{style}:{err}")

        if good is None:
            print("  NO_WORKING_PREFIX_URL")
            results.append({
                **base,
                "status": "NO_WORKING_PREFIX_URL",
                "prefix_len": n,
                "errors": " | ".join(errors),
            })
            continue

        url_style, good_url, data = good
        print(f"  prefix_len={n}; working={url_style}/CURL_STREAM")

        if not args.apply:
            print("  WOULD_REPAIR")
            results.append({
                **base,
                "status": "WOULD_REPAIR",
                "prefix_len": n,
                "working_url_style": url_style,
                "fetch_method": "CURL_STREAM",
            })
        else:
            safe_asset = "".join(
                c if c.isalnum() or c in "-_."
                else "_"
                for c in asset
            )

            backup = (
                backup_dir
                / f"{scene}__{safe_asset}__{n}bytes.bin"
            )

            with open(part, "rb") as f:
                old = f.read(n)

            if not backup.exists():
                backup.write_bytes(old)

            size_before = part.stat().st_size

            with open(part, "r+b") as f:
                f.seek(0)
                f.write(data)
                f.flush()

            with open(part, "rb") as f:
                verify = f.read(16)

            size_after = part.stat().st_size

            if is_tiff_magic(verify) and size_before == size_after:
                status = "REPAIRED"
                print("  ✅ REPAIRED")
            else:
                status = "REPAIR_VERIFY_FAILED"
                print("  REPAIR_VERIFY_FAILED")

            results.append({
                **base,
                "status": status,
                "prefix_len": n,
                "working_url_style": url_style,
                "fetch_method": "CURL_STREAM",
                "backup_path": str(backup),
                "size_before": size_before,
                "size_after": size_after,
            })

        if pos % 25 == 0:
            cp = state / "repair_checkpoint.csv"
            pd.DataFrame(results).to_csv(cp, index=False)
            print(f"  checkpoint -> {cp}")

        if pos < len(corrupt):
            time.sleep(args.sleep)

    out = pd.DataFrame(results)

    out_path = state / (
        "repair_results_apply.csv"
        if args.apply
        else "repair_results_dryrun.csv"
    )
    out.to_csv(out_path, index=False)

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(out["status"].value_counts(dropna=False).to_string())

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

    if "working_url_style" in out.columns:
        print()
        print("WORKING URL STYLES")
        print(
            out["working_url_style"]
            .dropna()
            .value_counts()
            .to_string()
        )

    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
