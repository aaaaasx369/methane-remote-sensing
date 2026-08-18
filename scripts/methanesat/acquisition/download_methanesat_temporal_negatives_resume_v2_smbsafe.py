#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_methanesat_temporal_negatives_resume.py

Resume-safe Phase B downloader for validated SAME-LOCATION temporal negatives.

Input
-----
01_strict_temporal_negatives.csv

For each row:
  - use the exact positive-source latitude/longitude
  - open the candidate MethaneSAT L3 collection
  - select XCH4
  - download a source-centered ~480 m crop at 45 m scale

Durability / resume guarantees
------------------------------
1. Existing valid .tif -> SKIP
2. New download -> writes to .part first
3. Validate .part with rasterio before accepting
4. fsync .part
5. atomic os.replace(.part, final.tif)
6. append one status JSONL row + fsync
7. rerunning the exact same command resumes; valid completed TIFFs are never redownloaded
8. network disconnect does not erase the local checkpoint
9. --restart-checkpoint clears only the local status log; it NEVER deletes TIFFs

The downloader deliberately does not convert these weak negatives into
"confirmed zero-emission" labels. It preserves validation_class/provenance.

Official EE asset:
  projects/edf-methanesat-ee/assets/public-preview/L3concentration
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Missing earthengine-api.\n"
        "Install with:\n"
        "  python -m pip install earthengine-api\n"
    ) from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing requests.\n"
        "Install with:\n"
        "  python -m pip install requests\n"
    ) from exc

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "Missing rasterio.\n"
        "Install with:\n"
        "  python -m pip install rasterio\n"
    ) from exc


L3_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L3concentration"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="~/methane_release_project/methanesat_temporal_negative_validation/01_strict_temporal_negatives.csv",
        help="Validated strict temporal-negative CSV.",
    )
    p.add_argument(
        "--out",
        default="/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/03_temporal_negatives",
        help="Directory for downloaded TIFFs and manifests.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="~/methane_release_project/methanesat_temporal_negative_download_checkpoints",
        help="LOCAL checkpoint directory. Keep this off SMB.",
    )
    p.add_argument(
        "--project",
        default="methane-release-gee",
        help="Earth Engine Cloud project.",
    )
    p.add_argument(
        "--crop-half-m",
        type=float,
        default=240.0,
        help="240 m gives a 480 m x 480 m source-centered region.",
    )
    p.add_argument(
        "--scale-m",
        type=float,
        default=45.0,
        help="Earth Engine download scale in meters.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=5,
        help="HTTP/EE retries per row.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP read timeout seconds.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Pause between successful downloads.",
    )
    p.add_argument(
        "--restart-checkpoint",
        action="store_true",
        help="Clear only local status logs. Existing valid TIFFs remain and will still be skipped.",
    )
    return p.parse_args()


def initialize_ee(project: str):
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed.\n"
            "Run:\n"
            "  earthengine authenticate\n"
            f"Then rerun with --project {project!r}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def sanitize(s: Any) -> str:
    s = str(s if s is not None else "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "unknown"


def norm_collection(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def stable_row_id(row: pd.Series) -> str:
    parts = [
        str(row.get("positive_sample_id", "")),
        norm_collection(row.get("candidate_collection_id", "")),
        f"{float(row.get('latitude')):.8f}",
        f"{float(row.get('longitude')):.8f}",
        str(row.get("candidate_time_start", "")),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def output_filename(row: pd.Series) -> str:
    pid = sanitize(row.get("positive_sample_id", "positive"))
    cid = sanitize(norm_collection(row.get("candidate_collection_id", "")))
    rank = row.get("post_validation_rank", row.get("selection_rank", ""))
    try:
        rank_s = f"r{int(float(rank)):02d}"
    except Exception:
        rank_s = "rNA"
    rid = stable_row_id(row)
    return f"{pid}__TEMPNEG__{rank_s}__c{cid}__{rid}.tif"


def append_jsonl_fsync(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {}
    for k, v in rec.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = None if np.isnan(v) else float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, pd.Timestamp):
            v = None if pd.isna(v) else v.isoformat()
        else:
            try:
                if pd.isna(v):
                    v = None
            except Exception:
                pass
        safe[str(k)] = v

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)



class NetworkShareDisconnected(RuntimeError):
    """Raised when the SMB/network output disappears or becomes unresponsive."""


def assert_output_share_alive(outdir: Path):
    outdir = Path(outdir)

    if str(outdir).startswith("/Volumes/"):
        parts = outdir.parts
        if len(parts) < 3:
            raise NetworkShareDisconnected(f"Invalid /Volumes path: {outdir}")

        mount_root = Path("/", "Volumes", parts[2])

        try:
            if not mount_root.exists():
                raise NetworkShareDisconnected(
                    f"SMB mount disappeared: {mount_root}"
                )
            if not outdir.exists():
                raise NetworkShareDisconnected(
                    f"Output path disappeared: {outdir}"
                )
        except (TimeoutError, OSError) as exc:
            raise NetworkShareDisconnected(
                f"SMB stat failed for {outdir}: {type(exc).__name__}: {exc}"
            ) from exc

    probe = outdir / ".smb_healthcheck.tmp"
    try:
        with probe.open("wb") as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
        probe.unlink(missing_ok=True)
    except (TimeoutError, OSError) as exc:
        raise NetworkShareDisconnected(
            f"SMB write probe failed for {outdir}: {type(exc).__name__}: {exc}"
        ) from exc


def ensure_output_mount(outdir: Path):
    s = str(outdir)
    if s.startswith("/Volumes/"):
        parts = outdir.parts
        if len(parts) < 3:
            raise RuntimeError(f"Invalid /Volumes output path: {outdir}")
        mount_root = Path("/", "Volumes", parts[2])
        if not mount_root.exists():
            raise RuntimeError(
                f"NETWORK SHARE DISCONNECTED: {mount_root}\n"
                "Reconnect the share, then rerun the SAME command.\n"
                "Local checkpoint remains safe."
            )

    outdir.mkdir(parents=True, exist_ok=True)
    assert_output_share_alive(outdir)


def validate_tiff(path: Path) -> tuple[bool, dict]:
    try:
        exists = path.exists()
        file_bytes = path.stat().st_size if exists else 0
    except (TimeoutError, OSError) as exc:
        raise NetworkShareDisconnected(
            f"SMB file stat failed: {path}\n{type(exc).__name__}: {exc}"
        ) from exc

    meta = {
        "file_bytes": file_bytes,
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "valid_pixel_fraction": None,
        "xch4_min": None,
        "xch4_median": None,
        "xch4_max": None,
        "crs": None,
    }

    if not exists or file_bytes <= 0:
        return False, meta

    try:
        with rasterio.open(path) as ds:
            meta["width"] = ds.width
            meta["height"] = ds.height
            meta["bands"] = ds.count
            meta["dtype"] = ds.dtypes[0] if ds.count else None
            meta["crs"] = str(ds.crs) if ds.crs else ""

            if ds.count < 1 or ds.width < 2 or ds.height < 2:
                return False, meta

            arr = ds.read(1, masked=True).astype("float64")
            vals = arr.compressed()
            if vals.size == 0:
                meta["valid_pixel_fraction"] = 0.0
                return False, meta

            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                meta["valid_pixel_fraction"] = 0.0
                return False, meta

            meta["valid_pixel_fraction"] = float(finite.size / arr.size)
            meta["xch4_min"] = float(np.min(finite))
            meta["xch4_median"] = float(np.median(finite))
            meta["xch4_max"] = float(np.max(finite))

            # Broad sanity only: XCH4 retrievals should be finite and not an all-constant broken file.
            if not np.isfinite(meta["xch4_median"]):
                return False, meta

            return True, meta

    except NetworkShareDisconnected:
        raise
    except (TimeoutError, OSError) as exc:
        raise NetworkShareDisconnected(
            f"SMB read failed: {path}\n{type(exc).__name__}: {exc}"
        ) from exc
    except Exception:
        return False, meta


def get_l3_image(collection_id: str) -> ee.Image:
    # Asset collection IDs are expected with leading c in the L3 public collection.
    cid_norm = norm_collection(collection_id)
    variants = [f"c{cid_norm}", cid_norm]

    ic0 = ee.ImageCollection(L3_ASSET)
    for cid in variants:
        ic = ic0.filter(ee.Filter.eq("collection_id", cid))
        n = int(ic.size().getInfo())
        if n > 0:
            return ee.Image(ic.first())

    raise RuntimeError(
        f"No MethaneSAT L3 image found for candidate collection {collection_id!r} "
        f"(tried {variants})"
    )


def build_download_url(row: pd.Series, crop_half_m: float, scale_m: float) -> str:
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    cid = str(row["candidate_collection_id"])

    image = get_l3_image(cid).select("XCH4")
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(crop_half_m).bounds()

    params = {
        "region": region,
        "scale": scale_m,
        "format": "GEO_TIFF",
    }
    return image.getDownloadURL(params)


def stream_download(url: str, part: Path, timeout: int):
    part.parent.mkdir(parents=True, exist_ok=True)

    # Always restart an incomplete .part from byte zero. The accepted final TIFF
    # is never touched until the new .part has validated.
    if part.exists():
        part.unlink()

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = str(r.headers.get("Content-Type", "")).lower()

        # EE error responses can occasionally be JSON/HTML even with HTTP 200-ish behavior.
        if "text/html" in ctype or "application/json" in ctype:
            preview = r.content[:1000]
            raise RuntimeError(
                f"Unexpected download Content-Type={ctype}; response={preview!r}"
            )

        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


def main():
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    outdir = Path(args.out).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status_jsonl = checkpoint_dir / "download_status.jsonl"
    if args.restart_checkpoint and status_jsonl.exists():
        status_jsonl.unlink()
        print("[RESTART] Local status log cleared.")
        print("[RESTART] Existing valid TIFF files will NOT be deleted.")

    ensure_output_mount(outdir)
    outdir = outdir.resolve()
    tif_dir = outdir / "tif"
    tif_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    required = [
        "positive_sample_id",
        "candidate_collection_id",
        "latitude",
        "longitude",
    ]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {missing}\n"
            f"Columns: {list(manifest.columns)}"
        )

    # Strict safeguard: only accept download-ready rows when that column exists.
    if "download_ready" in manifest.columns:
        ready = manifest["download_ready"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        manifest = manifest[ready].copy()

    if "validation_class" in manifest.columns:
        allowed = manifest["validation_class"].isin(
            ["CONFIRMED_NO_RELEASE", "TEMPORAL_WEAK_NEGATIVE_CLEAN"]
        )
        manifest = manifest[allowed].copy()

    manifest = manifest.reset_index(drop=True)
    initialize_ee(args.project)

    print("=" * 80)
    print("METHANESAT SAME-SITE TEMPORAL NEGATIVE DOWNLOADER — RESUME SAFE")
    print("=" * 80)
    print("Rows to process:", len(manifest))
    print("Output TIFF directory:", tif_dir)
    print("LOCAL checkpoint:", status_jsonl)
    print("Crop:", f"{2*args.crop_half_m:.0f} m x {2*args.crop_half_m:.0f} m")
    print("Scale:", f"{args.scale_m:g} m")
    print()

    session_rows = []
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, row in manifest.iterrows():
        assert_output_share_alive(outdir)

        rid = stable_row_id(row)
        fn = output_filename(row)
        final = tif_dir / fn
        part = final.with_suffix(final.suffix + ".part")

        prefix = f"[{i+1}/{len(manifest)}] {row.get('positive_sample_id')} c{norm_collection(row.get('candidate_collection_id'))}"

        # Resume path: trust only files that reopen successfully.
        valid_existing, existing_meta = validate_tiff(final)
        if valid_existing:
            skip_count += 1
            ok_count += 1
            print(prefix, "SKIP valid existing", fn)
            rec = {
                **row.to_dict(),
                "row_id": rid,
                "download_status": "SKIPPED_VALID_EXISTING",
                "output_tif": str(final),
                **existing_meta,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session_rows.append(rec)
            continue

        if final.exists():
            # Never silently overwrite an invalid final file. Preserve it for diagnosis.
            bad = final.with_suffix(final.suffix + f".invalid.{int(time.time())}")
            final.rename(bad)
            print(prefix, "existing TIFF invalid -> preserved as", bad.name)

        last_error = ""
        success = False
        meta = {}

        for attempt in range(1, args.retries + 1):
            try:
                print(prefix, f"download attempt {attempt}/{args.retries}")

                url = build_download_url(
                    row=row,
                    crop_half_m=args.crop_half_m,
                    scale_m=args.scale_m,
                )
                stream_download(url, part, timeout=args.timeout)

                valid, meta = validate_tiff(part)
                if not valid:
                    raise RuntimeError(
                        f"Downloaded file failed TIFF validation: {part}"
                    )

                # Atomic acceptance.
                os.replace(part, final)
                success = True
                break

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                print("Completed final TIFFs remain intact.")
                print("The current .part, if any, is not considered complete.")
                raise

            except NetworkShareDisconnected:
                raise

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print("    ERROR:", last_error)

                if part.exists():
                    try:
                        part.unlink()
                    except Exception:
                        pass

                if attempt < args.retries:
                    delay = min(60, 2 ** (attempt - 1) * 3)
                    print(f"    retrying in {delay}s...")
                    time.sleep(delay)

        if success:
            ok_count += 1
            print("    SAVED:", final)
            rec = {
                **row.to_dict(),
                "row_id": rid,
                "download_status": "DOWNLOADED",
                "output_tif": str(final),
                **meta,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session_rows.append(rec)
            time.sleep(args.sleep)
        else:
            fail_count += 1
            print("    FAILED after retries")
            rec = {
                **row.to_dict(),
                "row_id": rid,
                "download_status": "FAILED",
                "output_tif": str(final),
                "error": last_error,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session_rows.append(rec)

    # Re-audit every expected final file from disk.
    # On SMB, a single os.stat can time out even after downloads succeeded.
    # Treat that as a network problem, not a data failure.
    assert_output_share_alive(outdir)

    audit_rows = []
    for _, row in manifest.iterrows():
        final = tif_dir / output_filename(row)
        try:
            valid, meta = validate_tiff(final)
            try:
                final_exists = final.exists()
            except (TimeoutError, OSError) as exc:
                raise NetworkShareDisconnected(
                    f"SMB final-audit stat failed: {final}\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            audit_rows.append({
                **row.to_dict(),
                "row_id": stable_row_id(row),
                "expected_tif": str(final),
                "final_exists": final_exists,
                "final_valid": valid,
                **meta,
            })

        except NetworkShareDisconnected as exc:
            print()
            print("=" * 80)
            print("SMB / NETWORK SHARE UNRESPONSIVE DURING FINAL AUDIT")
            print("=" * 80)
            print(str(exc))
            print()
            print("Downloads completed before this point are still intact.")
            print("The program is stopping intentionally instead of marking files as failed.")
            print("Reconnect the SAME SMB path and rerun the SAME command.")
            print("Valid existing TIFFs will be skipped; the final audit will run again.")
            raise SystemExit(3)

    audit = pd.DataFrame(audit_rows)
    atomic_csv(audit, outdir / "00_download_audit.csv")

    good = audit[audit["final_valid"].eq(True)].copy()
    atomic_csv(good, outdir / "01_downloaded_temporal_negatives.csv")

    failed = audit[~audit["final_valid"].eq(True)].copy()
    atomic_csv(failed, outdir / "02_missing_or_invalid.csv")

    summary = [
        "# MethaneSAT same-site temporal-negative download",
        "",
        "## Requested",
        f"- Manifest rows: {len(manifest)}",
        f"- Crop: {2*args.crop_half_m:.0f} m × {2*args.crop_half_m:.0f} m",
        f"- Download scale: {args.scale_m:g} m",
        "",
        "## Final on-disk audit",
        f"- Valid TIFFs: {len(good)} / {len(manifest)}",
        f"- Missing or invalid: {len(failed)}",
        f"- Unique positive sources represented: {good['positive_sample_id'].nunique() if len(good) else 0}",
        "",
        "## Session",
        f"- Downloaded this run: {sum(r.get('download_status') == 'DOWNLOADED' for r in session_rows)}",
        f"- Skipped valid existing this run: {sum(r.get('download_status') == 'SKIPPED_VALID_EXISTING' for r in session_rows)}",
        f"- Failed this run: {sum(r.get('download_status') == 'FAILED' for r in session_rows)}",
        "",
        "## Resume policy",
        "- Valid existing TIFFs are skipped.",
        "- Incomplete downloads use .part and are never accepted as final TIFFs.",
        "- Rerun the exact same command after interruption.",
        "- Local checkpoint is intentionally separate from the network output.",
        "",
        "## Label caveat",
        "- TEMPORAL_WEAK_NEGATIVE_CLEAN means no known positive/release record was found under the validation rules.",
        "- It is not proof of physically zero methane emissions.",
    ]
    (outdir / "SUMMARY_DOWNLOAD.md").write_text("\n".join(summary), encoding="utf-8")

    print()
    print("=" * 80)
    print("FINAL AUDIT")
    print("=" * 80)
    print(f"Valid TIFFs       : {len(good)}/{len(manifest)}")
    print(f"Missing / invalid : {len(failed)}")
    print(
        "Positive sources  :",
        good["positive_sample_id"].nunique() if len(good) else 0,
    )
    print()
    print("Outputs:")
    for fn in [
        "SUMMARY_DOWNLOAD.md",
        "00_download_audit.csv",
        "01_downloaded_temporal_negatives.csv",
        "02_missing_or_invalid.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    try:
        main()
    except NetworkShareDisconnected as exc:
        print()
        print("=" * 80)
        print("SMB / NETWORK SHARE DISCONNECTED")
        print("=" * 80)
        print(str(exc))
        print()
        print("Download stopped intentionally.")
        print("Completed valid TIFFs remain intact.")
        print("Reconnect the SAME SMB mount path, then rerun the SAME command.")
        print("Existing valid TIFFs will be skipped.")
        raise SystemExit(3)
