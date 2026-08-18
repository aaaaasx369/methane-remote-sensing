#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import time
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--builder",
        default="~/Downloads/build_methanesat_120_paired_image_benchmark_v2_smbsafe.py",
    )
    p.add_argument(
        "--failed",
        default=(
            "/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/"
            "05_paired_image_benchmark_120/manifests/04_failed_pairs.csv"
        ),
    )
    p.add_argument(
        "--staging",
        default="~/methane_release_project/methanesat_120_failed_side_recovery_staging",
    )
    p.add_argument("--project", default="methane-release-gee")
    p.add_argument("--crop-half-m", type=float, default=240.0)
    p.add_argument("--scale-m", type=float, default=45.0)
    p.add_argument("--min-valid-fraction", type=float, default=0.50)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=180)
    return p.parse_args()


def import_builder(path: Path):
    spec = importlib.util.spec_from_file_location("msat_pair_builder_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import builder: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atomic_copy_to_remote(local: Path, remote: Path, builder, min_valid_fraction: float):
    remote.parent.mkdir(parents=True, exist_ok=True)
    benchmark_root = remote.parents[2]
    builder.assert_output_share_alive(benchmark_root)

    tmp = remote.with_suffix(remote.suffix + ".recovery.part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    with local.open("rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())

    builder.assert_output_share_alive(benchmark_root)

    ok, meta, _, _ = builder.validate_tiff(tmp, min_valid_fraction)
    if not ok:
        raise RuntimeError(f"Remote staged copy failed QA: {tmp}; meta={meta}")

    os.replace(tmp, remote)
    builder.assert_output_share_alive(benchmark_root)


def main():
    args = parse_args()

    builder_path = Path(args.builder).expanduser()
    failed_path = Path(args.failed).expanduser()
    staging = Path(args.staging).expanduser()
    staging.mkdir(parents=True, exist_ok=True)

    if not builder_path.exists():
        raise FileNotFoundError(f"Builder not found: {builder_path}")
    if not failed_path.exists():
        raise FileNotFoundError(f"Failed CSV not found: {failed_path}")

    b = import_builder(builder_path)
    b.initialize_ee(args.project)

    df = pd.read_csv(failed_path, low_memory=False)

    required = [
        "pair_id", "latitude", "longitude",
        "positive_collection_id", "candidate_collection_id",
        "positive_tif_valid", "control_tif_valid",
        "positive_tif", "temporal_control_tif",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    jobs = []
    for _, r in df.iterrows():
        if str(r["positive_tif_valid"]).lower() not in {"true", "1"}:
            jobs.append({
                "pair_id": r["pair_id"],
                "side": "POSITIVE",
                "collection_id": b.norm_collection(r["positive_collection_id"]),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "remote": Path(str(r["positive_tif"])),
            })

        if str(r["control_tif_valid"]).lower() not in {"true", "1"}:
            jobs.append({
                "pair_id": r["pair_id"],
                "side": "CONTROL",
                "collection_id": b.norm_collection(r["candidate_collection_id"]),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "remote": Path(str(r["temporal_control_tif"])),
            })

    print("=" * 88)
    print("METHANESAT TARGETED FAILED-SIDE RECOVERY")
    print("=" * 88)
    print("Failed pairs :", len(df))
    print("Missing sides:", len(jobs))
    print("Local staging:", staging)

    results = []

    for i, job in enumerate(jobs, 1):
        pair_id = job["pair_id"]
        side = job["side"]
        cid = job["collection_id"]
        lat = job["lat"]
        lon = job["lon"]
        remote = job["remote"]

        local = staging / f"{pair_id}__{side}__c{cid}.tif"
        part = local.with_suffix(".tif.part")

        print()
        print("-" * 88)
        print(f"[{i}/{len(jobs)}] {pair_id} {side}")
        print("collection:", cid)
        print("location  :", f"{lat:.7f}", f"{lon:.7f}")

        try:
            remote_ok, remote_meta, _, _ = b.validate_tiff(
                remote, args.min_valid_fraction
            )
        except Exception:
            remote_ok = False
            remote_meta = {}

        if remote_ok:
            print("REMOTE ALREADY VALID -> SKIP")
            results.append({
                **job,
                "status": "REMOTE_ALREADY_VALID",
                "valid_fraction": remote_meta.get("valid_pixel_fraction"),
                "error": "",
            })
            continue

        success = False
        last_error = ""
        local_meta = {}

        for attempt in range(1, args.retries + 1):
            try:
                print(f"LOCAL download attempt {attempt}/{args.retries}")

                url, matched_cid, props = b.build_download_url(
                    cid, lat, lon, args.crop_half_m, args.scale_m
                )
                b.stream_download(url, part, args.timeout)

                ok, local_meta, _, _ = b.validate_tiff(
                    part, args.min_valid_fraction
                )

                print(
                    "  local:",
                    "PASS" if ok else "QA_FAIL",
                    "valid_fraction=",
                    local_meta.get("valid_pixel_fraction"),
                )

                if not ok:
                    last_error = (
                        "LOCAL_QA_FAIL: "
                        f"valid_fraction={local_meta.get('valid_pixel_fraction')}"
                    )
                    if part.exists():
                        part.unlink()
                    break

                os.replace(part, local)
                success = True
                break

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print("  ERROR:", last_error)

                if part.exists():
                    try:
                        part.unlink()
                    except Exception:
                        pass

                if attempt < args.retries:
                    wait = min(60, 3 * (2 ** (attempt - 1)))
                    print(f"  retry in {wait}s")
                    time.sleep(wait)

        if not success:
            print("RESULT: NOT RECOVERED")
            results.append({
                **job,
                "status": "NOT_RECOVERED",
                "valid_fraction": local_meta.get("valid_pixel_fraction"),
                "error": last_error,
            })
            continue

        print("LOCAL VALID:", local)
        print("COPY TO SMB:", remote)

        try:
            atomic_copy_to_remote(
                local, remote, b, args.min_valid_fraction
            )

            remote_ok, remote_meta, _, _ = b.validate_tiff(
                remote, args.min_valid_fraction
            )
            if not remote_ok:
                raise RuntimeError("Final remote TIFF failed validation")

            print(
                "RESULT: RECOVERED",
                f"valid_fraction={remote_meta.get('valid_pixel_fraction')}",
            )

            results.append({
                **job,
                "status": "RECOVERED",
                "valid_fraction": remote_meta.get("valid_pixel_fraction"),
                "error": "",
            })

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print("SMB COPY FAILED:", last_error)
            print("Validated local TIFF remains at:", local)
            results.append({
                **job,
                "status": "LOCAL_VALID_SMB_COPY_FAILED",
                "valid_fraction": local_meta.get("valid_pixel_fraction"),
                "error": last_error,
            })
            break

    out_csv = staging / "RECOVERY_RESULTS.csv"
    rr = pd.DataFrame(results)
    rr.to_csv(out_csv, index=False)

    print()
    print("=" * 88)
    print("RECOVERY SUMMARY")
    print("=" * 88)
    if len(rr):
        print(rr["status"].value_counts().to_string())
    else:
        print("No jobs.")
    print("Results:", out_csv)


if __name__ == "__main__":
    main()
