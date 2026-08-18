#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repair_methanesat_120_with_next_best_controls.py

Targeted repair for the MethaneSAT 120-pair benchmark.

What it does
------------
1) Reads the 7 failed pairs from the previous 120-pair build.
2) Permanently excludes pairs whose POSITIVE L3 crop failed the fixed QA threshold.
3) For each pair whose positive passed but selected CONTROL failed:
   - goes back to 02_eligible_negative_candidates.csv
   - excludes the failed control collection
   - preserves the original ranking logic
   - tries the next-best candidates one by one
   - downloads each candidate to LOCAL Mac staging first
   - validates the exact 480 m / 45 m XCH4 crop locally
   - accepts the first candidate with valid fraction >= 0.50
4) Mirrors only the accepted replacement control to SMB.
5) Builds the missing control sample NPZ and pair NPZ using the SAME preprocessing
   functions as the original paired benchmark builder.
6) Writes a repaired canonical primary-pair manifest.

It does NOT lower the QA threshold.
It does NOT alter the 113 pairs that already passed.
It does NOT call no-detection controls "confirmed zero emission".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTECTED_POSITIVE_COLUMNS = {
    "pair_id",
    "positive_id",
    "positive_source",
    "positive_sample_id",
    "positive_collection_id",
    "positive_time",
    "latitude",
    "longitude",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--search-dir",
        default=(
            "~/methane_release_project/"
            "methanesat_156_far_temporal_negative_search"
        ),
        help="Folder containing 02_eligible_negative_candidates.csv and 03_best_one_negative_per_positive.csv",
    )
    p.add_argument(
        "--benchmark-dir",
        default=(
            "/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/"
            "05_paired_image_benchmark_120"
        ),
    )
    p.add_argument(
        "--builder",
        default=(
            "~/Downloads/"
            "build_methanesat_120_paired_image_benchmark_v2_smbsafe.py"
        ),
        help="Existing paired builder; imported to guarantee identical download/QA/preprocessing.",
    )
    p.add_argument(
        "--staging",
        default=(
            "~/methane_release_project/"
            "methanesat_120_next_best_control_recovery"
        ),
        help="LOCAL Mac staging/checkpoint folder.",
    )
    p.add_argument("--project", default="methane-release-gee")

    p.add_argument("--crop-half-m", type=float, default=240.0)
    p.add_argument("--scale-m", type=float, default=45.0)
    p.add_argument("--min-valid-fraction", type=float, default=0.50)
    p.add_argument("--npz-size", type=int, default=224)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)

    return p.parse_args()


def import_builder(path: Path):
    spec = importlib.util.spec_from_file_location("msat_pair_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import builder: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def clean(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def atomic_local_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: conv(v) for k, v in rec.items()},
                           ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_copy_file(local: Path, remote: Path, benchmark_root: Path, b):
    b.assert_output_share_alive(benchmark_root)
    remote.parent.mkdir(parents=True, exist_ok=True)

    tmp = remote.with_suffix(remote.suffix + ".replacement.part")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    with local.open("rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())

    b.assert_output_share_alive(benchmark_root)
    os.replace(tmp, remote)
    b.assert_output_share_alive(benchmark_root)


def sorted_candidates(sub: pd.DataFrame) -> pd.DataFrame:
    x = sub.copy()

    for c in [
        "negative_evidence_rank",
        "time_rank",
        "abs_delta_days",
        "local_valid_fraction",
    ]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    return x.sort_values(
        [
            "negative_evidence_rank",
            "time_rank",
            "abs_delta_days",
            "local_valid_fraction",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_replacement_control_sample(
    *,
    pair_id: str,
    candidate_row: pd.Series,
    local_tif: Path,
    remote_tif: Path,
    control_npz: Path,
    positive_npz: Path,
    pair_npz: Path,
    args,
    b,
):
    ok, meta, arr, valid = b.validate_tiff(
        local_tif, args.min_valid_fraction
    )
    if not ok:
        raise RuntimeError(
            f"Local validated replacement unexpectedly failed QA: {local_tif}"
        )

    x224, mask224 = b.resize_float_with_mask(
        arr, valid, args.npz_size
    )
    if float(mask224.mean()) < args.min_valid_fraction:
        raise RuntimeError(
            f"Replacement standardized valid fraction {float(mask224.mean()):.4f} "
            f"< {args.min_valid_fraction:.4f}"
        )

    evidence = clean(candidate_row.get("negative_evidence_tier"))
    cid = b.norm_collection(candidate_row["candidate_collection_id"])
    lat = float(candidate_row["latitude"])
    lon = float(candidate_row["longitude"])
    ctime = clean(candidate_row.get("candidate_time_start"))
    ptime = clean(candidate_row.get("positive_time"))

    b.atomic_npz(
        control_npz,
        xch4=x224.astype(np.float32),
        valid_mask=mask224.astype(np.uint8),
        binary_label=np.array(0, dtype=np.int8),
        class_name=np.array("temporal_control"),
        pair_id=np.array(pair_id),
        collection_id=np.array(cid),
        latitude=np.array(lat, dtype=np.float64),
        longitude=np.array(lon, dtype=np.float64),
        acquisition_time=np.array(ctime),
        control_evidence_tier=np.array(evidence),
    )

    if not positive_npz.exists():
        raise RuntimeError(
            f"Existing positive NPZ is missing for repairable pair: {positive_npz}"
        )

    with np.load(positive_npz, allow_pickle=False) as pz:
        pos_x = pz["xch4"].copy()
        pos_mask = pz["valid_mask"].copy()

    b.atomic_npz(
        pair_npz,
        positive_xch4=pos_x.astype(np.float32),
        temporal_control_xch4=x224.astype(np.float32),
        positive_valid_mask=pos_mask.astype(np.uint8),
        temporal_control_valid_mask=mask224.astype(np.uint8),
        pair_id=np.array(pair_id),
        latitude=np.array(lat, dtype=np.float64),
        longitude=np.array(lon, dtype=np.float64),
        positive_collection_id=np.array(
            b.norm_collection(candidate_row["positive_collection_id"])
        ),
        temporal_control_collection_id=np.array(cid),
        positive_time=np.array(ptime),
        temporal_control_time=np.array(ctime),
        abs_delta_days=np.array(
            float(candidate_row["abs_delta_days"]),
            dtype=np.float32,
        ),
        control_evidence_tier=np.array(evidence),
    )

    features = b.image_features(
        x224,
        mask224,
        crop_size_m=2 * args.crop_half_m,
    )

    return meta, features


def main():
    args = parse_args()

    search_dir = Path(args.search_dir).expanduser()
    benchmark_dir = Path(args.benchmark_dir).expanduser()
    builder_path = Path(args.builder).expanduser()
    staging = Path(args.staging).expanduser()
    staging.mkdir(parents=True, exist_ok=True)

    eligible_path = search_dir / "02_eligible_negative_candidates.csv"
    frozen_path = benchmark_dir / "manifests" / "00_input_pairs_frozen.csv"
    failed_path = benchmark_dir / "manifests" / "04_failed_pairs.csv"
    valid_path = benchmark_dir / "manifests" / "02_valid_pairs.csv"
    features_path = benchmark_dir / "manifests" / "03_sample_image_features.csv"

    for p in [eligible_path, frozen_path, failed_path, valid_path, builder_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    b = import_builder(builder_path)
    b.initialize_ee(args.project)
    b.assert_output_share_alive(benchmark_dir)

    eligible = pd.read_csv(eligible_path, low_memory=False)
    frozen = pd.read_csv(frozen_path, low_memory=False)
    failed = pd.read_csv(failed_path, low_memory=False)
    valid113 = pd.read_csv(valid_path, low_memory=False)

    if len(frozen) != 120:
        raise RuntimeError(
            f"Expected original frozen manifest to contain 120 rows, got {len(frozen)}"
        )

    positive_fail = failed[
        ~failed["positive_tif_valid"].map(as_bool)
    ].copy()

    control_fail = failed[
        failed["positive_tif_valid"].map(as_bool)
        & ~failed["control_tif_valid"].map(as_bool)
    ].copy()

    print("=" * 92)
    print("METHANESAT NEXT-BEST CONTROL REPAIR")
    print("=" * 92)
    print("Previously valid pairs :", len(valid113))
    print("Positive QA exclusions :", len(positive_fail))
    print("Control replacements   :", len(control_fail))
    print("Fixed QA threshold     :", args.min_valid_fraction)
    print()
    print("Positive-QA exclusions:")
    for _, r in positive_fail.iterrows():
        print(
            " ",
            r["pair_id"],
            b.norm_collection(r["positive_collection_id"]),
            "valid=",
            r.get("positive_valid_fraction"),
        )

    replacement_records = []
    replacement_feature_rows = []
    repaired_rows_by_pair = {}

    checkpoint = staging / "replacement_attempts.jsonl"

    for j, (_, fr) in enumerate(control_fail.iterrows(), 1):
        pair_id = clean(fr["pair_id"])
        positive_id = clean(fr["positive_id"])
        failed_cid = b.norm_collection(fr["candidate_collection_id"])
        lat = float(fr["latitude"])
        lon = float(fr["longitude"])

        sub = eligible[
            eligible["positive_id"].astype(str).eq(positive_id)
        ].copy()

        sub["candidate_collection_id_norm"] = (
            sub["candidate_collection_id"].map(b.norm_collection)
        )

        sub = sub[
            ~sub["candidate_collection_id_norm"].eq(failed_cid)
        ].copy()

        # Safety: all replacement candidates must still be eligible.
        if "candidate_status" in sub.columns:
            sub = sub[sub["candidate_status"].eq("ELIGIBLE")].copy()

        sub = sorted_candidates(sub)

        print()
        print("-" * 92)
        print(f"[{j}/{len(control_fail)}] {pair_id}")
        print("positive_id    :", positive_id)
        print("failed control :", failed_cid)
        print("alternatives   :", len(sub))

        recovered = False

        for rank, (_, cand) in enumerate(sub.iterrows(), 1):
            cid = b.norm_collection(cand["candidate_collection_id"])

            # Exact same location sanity check.
            clat = float(cand["latitude"])
            clon = float(cand["longitude"])
            if abs(clat - lat) > 1e-8 or abs(clon - lon) > 1e-8:
                rec = {
                    "pair_id": pair_id,
                    "positive_id": positive_id,
                    "candidate_rank_tried": rank,
                    "candidate_collection_id": cid,
                    "status": "REJECT_COORDINATE_MISMATCH",
                    "local_valid_fraction_exact_download": None,
                    "error": f"candidate ({clat},{clon}) != pair ({lat},{lon})",
                }
                replacement_records.append(rec)
                append_jsonl(checkpoint, rec)
                continue

            local_tif = (
                staging / "local_tif" /
                f"{pair_id}__candidate{rank:02d}__c{cid}.tif"
            )
            local_tif.parent.mkdir(parents=True, exist_ok=True)
            part = local_tif.with_suffix(".tif.part")

            print(
                f"  candidate {rank:02d}: c{cid}",
                f"|Δt|={cand.get('abs_delta_days')}",
                f"tier={cand.get('negative_evidence_tier')}",
                f"search_QA={cand.get('local_valid_fraction')}",
            )

            # Reuse local staged TIFF only if it passes exact current QA.
            try:
                local_ok, local_meta, _, _ = b.validate_tiff(
                    local_tif, args.min_valid_fraction
                )
            except Exception:
                local_ok = False
                local_meta = {}

            if not local_ok:
                last_error = ""

                for attempt in range(1, args.retries + 1):
                    try:
                        url, matched_cid, props = b.build_download_url(
                            cid,
                            lat,
                            lon,
                            args.crop_half_m,
                            args.scale_m,
                        )
                        b.stream_download(url, part, args.timeout)

                        local_ok, local_meta, _, _ = b.validate_tiff(
                            part, args.min_valid_fraction
                        )

                        if not local_ok:
                            last_error = (
                                "EXACT_LOCAL_QA_FAIL: "
                                f"valid_fraction="
                                f"{local_meta.get('valid_pixel_fraction')}"
                            )
                            try:
                                part.unlink()
                            except Exception:
                                pass
                            # Deterministic data QA failure: no reason to retry
                            # this same acquisition.
                            break

                        os.replace(part, local_tif)
                        last_error = ""
                        break

                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        try:
                            if part.exists():
                                part.unlink()
                        except Exception:
                            pass
                        if attempt < args.retries:
                            time.sleep(min(30, 3 * (2 ** (attempt - 1))))

                if not local_ok:
                    vf = local_meta.get("valid_pixel_fraction")
                    print("     FAIL exact QA:", vf, last_error)
                    rec = {
                        "pair_id": pair_id,
                        "positive_id": positive_id,
                        "failed_original_control": failed_cid,
                        "candidate_rank_tried": rank,
                        "candidate_collection_id": cid,
                        "candidate_time_start": cand.get("candidate_time_start"),
                        "abs_delta_days": cand.get("abs_delta_days"),
                        "negative_evidence_tier": cand.get("negative_evidence_tier"),
                        "search_local_valid_fraction": cand.get("local_valid_fraction"),
                        "local_valid_fraction_exact_download": vf,
                        "status": "REJECT_EXACT_IMAGE_QA",
                        "error": last_error,
                    }
                    replacement_records.append(rec)
                    append_jsonl(checkpoint, rec)
                    continue

            exact_vf = float(local_meta["valid_pixel_fraction"])
            print("     PASS exact QA:", exact_vf)

            # Build the remote filenames using the existing pair slot ID.
            remote_ctrl_tif = (
                benchmark_dir / "raw_tif" / "temporal_control" /
                f"{pair_id}__CTRL__c{cid}.tif"
            )
            remote_ctrl_npz = (
                benchmark_dir / "npz" / "samples" /
                f"{pair_id}__CTRL__c{cid}.npz"
            )

            positive_npz = Path(clean(fr["positive_npz"]))
            pair_npz = Path(clean(fr["pair_npz"]))
            if not clean(fr["pair_npz"]):
                pair_npz = benchmark_dir / "npz" / "pairs" / f"{pair_id}.npz"

            try:
                atomic_copy_file(
                    local_tif,
                    remote_ctrl_tif,
                    benchmark_dir,
                    b,
                )

                remote_ok, remote_meta, _, _ = b.validate_tiff(
                    remote_ctrl_tif,
                    args.min_valid_fraction,
                )
                if not remote_ok:
                    raise RuntimeError(
                        "Remote replacement TIFF failed validation after copy"
                    )

                meta, feats = build_replacement_control_sample(
                    pair_id=pair_id,
                    candidate_row=cand,
                    local_tif=local_tif,
                    remote_tif=remote_ctrl_tif,
                    control_npz=remote_ctrl_npz,
                    positive_npz=positive_npz,
                    pair_npz=pair_npz,
                    args=args,
                    b=b,
                )

                repaired = frozen[
                    frozen["pair_id"].astype(str).eq(pair_id)
                ].copy()
                if len(repaired) != 1:
                    raise RuntimeError(
                        f"Could not uniquely locate {pair_id} in frozen manifest"
                    )

                repaired = repaired.iloc[0].copy()

                # Preserve positive identity fields; replace candidate/evidence
                # fields with the accepted alternative candidate.
                for col in frozen.columns:
                    if col in PROTECTED_POSITIVE_COLUMNS:
                        continue
                    if col in cand.index:
                        repaired[col] = cand[col]

                repaired["pair_id"] = pair_id
                repaired["replacement_control"] = True
                repaired["original_failed_candidate_collection_id"] = failed_cid
                repaired["replacement_candidate_rank"] = rank
                repaired["replacement_exact_valid_fraction"] = exact_vf
                repaired["temporal_control_tif"] = str(remote_ctrl_tif)
                repaired["temporal_control_npz"] = str(remote_ctrl_npz)
                repaired["pair_npz"] = str(pair_npz)

                repaired_rows_by_pair[pair_id] = repaired

                replacement_feature_rows.append({
                    "pair_id": pair_id,
                    "positive_id": positive_id,
                    "positive_sample_id": cand.get("positive_sample_id"),
                    "latitude": lat,
                    "longitude": lon,
                    "abs_delta_days": cand.get("abs_delta_days"),
                    "control_evidence_tier": cand.get("negative_evidence_tier"),
                    "class_name": "temporal_control",
                    "binary_label": 0,
                    "collection_id": cid,
                    "acquisition_time": cand.get("candidate_time_start"),
                    "sample_ok": True,
                    "valid_pixel_fraction_raw": exact_vf,
                    "valid_fraction_224": feats.get("valid_fraction_224"),
                    "mean": feats.get("mean"),
                    "median": feats.get("median"),
                    "p90": feats.get("p90"),
                    "p95": feats.get("p95"),
                    "p99": feats.get("p99"),
                    "max": feats.get("max"),
                    "center_r60_mean": feats.get("center_r60_mean"),
                    "ring_r120_220_median": feats.get(
                        "ring_r120_220_median"
                    ),
                    "center_minus_ring": feats.get("center_minus_ring"),
                })

                rec = {
                    "pair_id": pair_id,
                    "positive_id": positive_id,
                    "failed_original_control": failed_cid,
                    "candidate_rank_tried": rank,
                    "candidate_collection_id": cid,
                    "candidate_time_start": cand.get("candidate_time_start"),
                    "abs_delta_days": cand.get("abs_delta_days"),
                    "negative_evidence_tier": cand.get("negative_evidence_tier"),
                    "search_local_valid_fraction": cand.get("local_valid_fraction"),
                    "local_valid_fraction_exact_download": exact_vf,
                    "status": "RECOVERED_NEXT_BEST_CONTROL",
                    "remote_control_tif": str(remote_ctrl_tif),
                    "remote_control_npz": str(remote_ctrl_npz),
                    "pair_npz": str(pair_npz),
                    "error": "",
                }
                replacement_records.append(rec)
                append_jsonl(checkpoint, rec)

                print("     RECOVERED:", cid)
                recovered = True
                break

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                rec = {
                    "pair_id": pair_id,
                    "positive_id": positive_id,
                    "failed_original_control": failed_cid,
                    "candidate_rank_tried": rank,
                    "candidate_collection_id": cid,
                    "candidate_time_start": cand.get("candidate_time_start"),
                    "abs_delta_days": cand.get("abs_delta_days"),
                    "negative_evidence_tier": cand.get("negative_evidence_tier"),
                    "search_local_valid_fraction": cand.get("local_valid_fraction"),
                    "local_valid_fraction_exact_download": exact_vf,
                    "status": "LOCAL_PASS_REMOTE_OR_NPZ_FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                replacement_records.append(rec)
                append_jsonl(checkpoint, rec)
                print("     STOP:", rec["error"])
                print(
                    "     Local validated candidate remains at:",
                    local_tif,
                )
                # A remote failure is infrastructure, not a candidate failure.
                # Stop the full repair instead of moving to a weaker candidate.
                raise

        if not recovered:
            print("  NO REPLACEMENT PASSED for", pair_id)

    # ------------------------------------------------------------------
    # Build canonical repaired pair manifest.
    # Start from original frozen 120 rows, retain the 113 already-valid,
    # add successfully repaired control-fail pairs, and exclude the
    # positive-QA failures / unrecovered control failures.
    # ------------------------------------------------------------------

    valid_pair_ids = set(valid113["pair_id"].astype(str))
    repaired_pair_ids = set(repaired_rows_by_pair)

    final_rows = []

    for _, row in frozen.iterrows():
        pid = str(row["pair_id"])

        if pid in valid_pair_ids:
            rr = row.copy()
            rr["replacement_control"] = False
            rr["original_failed_candidate_collection_id"] = ""
            rr["replacement_candidate_rank"] = np.nan
            rr["replacement_exact_valid_fraction"] = np.nan
            final_rows.append(rr)
        elif pid in repaired_pair_ids:
            final_rows.append(repaired_rows_by_pair[pid])

    final_pairs = pd.DataFrame(final_rows)

    # Preserve original frozen order.
    order = {str(pid): i for i, pid in enumerate(frozen["pair_id"].astype(str))}
    if len(final_pairs):
        final_pairs["_order"] = final_pairs["pair_id"].astype(str).map(order)
        final_pairs = (
            final_pairs.sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    excluded_ids = set(frozen["pair_id"].astype(str)) - set(
        final_pairs["pair_id"].astype(str)
    )
    excluded = frozen[
        frozen["pair_id"].astype(str).isin(excluded_ids)
    ].copy()

    positive_fail_ids = set(positive_fail["pair_id"].astype(str))
    control_fail_ids = set(control_fail["pair_id"].astype(str))

    def reason(pid):
        if pid in positive_fail_ids:
            return "EXCLUDE_POSITIVE_L3_QA_FAIL"
        if pid in control_fail_ids and pid not in repaired_pair_ids:
            return "EXCLUDE_NO_REPLACEMENT_CONTROL_PASSED"
        return "EXCLUDED_OTHER"

    if len(excluded):
        excluded["exclusion_reason"] = excluded["pair_id"].astype(str).map(reason)

    attempts = pd.DataFrame(replacement_records)

    # Repaired image-feature table = existing features for final pairs,
    # with failed-control rows replaced by the accepted replacement rows.
    if features_path.exists():
        old_features = pd.read_csv(features_path, low_memory=False)
    else:
        old_features = pd.DataFrame()

    if len(old_features):
        final_id_set = set(final_pairs["pair_id"].astype(str))
        f = old_features[
            old_features["pair_id"].astype(str).isin(final_id_set)
        ].copy()

        repaired_ids = set(repaired_rows_by_pair)
        f = f[
            ~(
                f["pair_id"].astype(str).isin(repaired_ids)
                & f["class_name"].astype(str).eq("temporal_control")
            )
        ].copy()

        if replacement_feature_rows:
            f = pd.concat(
                [f, pd.DataFrame(replacement_feature_rows)],
                ignore_index=True,
                sort=False,
            )

        repaired_features = f
    else:
        repaired_features = pd.DataFrame(replacement_feature_rows)

    # Local copies first.
    local_attempts = staging / "05_replacement_control_attempts.csv"
    local_final = staging / "06_repaired_primary_pairs.csv"
    local_excluded = staging / "07_excluded_pairs.csv"
    local_features = staging / "08_repaired_sample_image_features.csv"

    atomic_local_csv(attempts, local_attempts)
    atomic_local_csv(final_pairs, local_final)
    atomic_local_csv(excluded, local_excluded)
    atomic_local_csv(repaired_features, local_features)

    # Mirror small canonical outputs to SMB.
    b.assert_output_share_alive(benchmark_dir)
    manifest_dir = benchmark_dir / "manifests"

    b.atomic_csv(
        attempts,
        manifest_dir / "05_replacement_control_attempts.csv",
    )
    b.atomic_csv(
        final_pairs,
        manifest_dir / "06_repaired_primary_pairs.csv",
    )
    b.atomic_csv(
        excluded,
        manifest_dir / "07_excluded_pairs.csv",
    )
    b.atomic_csv(
        repaired_features,
        manifest_dir / "08_repaired_sample_image_features.csv",
    )

    recovered_n = len(repaired_rows_by_pair)
    final_n = len(final_pairs)

    summary = [
        "# MethaneSAT repaired paired benchmark",
        "",
        f"- Original frozen pairs: 120",
        f"- Previously valid pairs: {len(valid113)}",
        f"- Positive-side L3 QA exclusions: {len(positive_fail)}",
        f"- Control-fail pairs searched for next-best alternatives: {len(control_fail)}",
        f"- Control pairs recovered with next-best exact-image-QA candidate: {recovered_n}",
        f"- Final primary paired benchmark: {final_n} positive + {final_n} temporal controls",
        f"- Fixed exact-image QA threshold: valid fraction >= {args.min_valid_fraction:.2f}",
        "",
        "No QA threshold was lowered.",
        "Temporal controls remain same-site, far-time no-detection controls unless independently confirmed otherwise.",
        "",
        "Canonical manifest:",
        "manifests/06_repaired_primary_pairs.csv",
        "",
        "Canonical feature table:",
        "manifests/08_repaired_sample_image_features.csv",
    ]

    summary_text = "\n".join(summary) + "\n"

    local_summary = staging / "SUMMARY_REPAIRED_BENCHMARK.md"
    local_summary.write_text(summary_text, encoding="utf-8")

    remote_summary = benchmark_dir / "SUMMARY_REPAIRED_BENCHMARK.md"
    tmp_summary = remote_summary.with_suffix(".md.tmp")
    tmp_summary.write_text(summary_text, encoding="utf-8")
    with tmp_summary.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_summary, remote_summary)

    print()
    print("=" * 92)
    print("FINAL REPAIR SUMMARY")
    print("=" * 92)
    print("Previously valid pairs :", len(valid113))
    print("Positive QA exclusions :", len(positive_fail))
    print("Control pairs recovered:", recovered_n, "/", len(control_fail))
    print("FINAL PRIMARY PAIRS    :", final_n)
    print()
    if recovered_n == len(control_fail):
        print(
            f"TARGET REACHED: {final_n} positives + "
            f"{final_n} same-site far-time controls"
        )
    else:
        print(
            "Some control-fail pairs had no alternative candidate that passed "
            "the exact image QA."
        )
    print()
    print("Canonical manifest:")
    print(" ", manifest_dir / "06_repaired_primary_pairs.csv")
    print("Feature table:")
    print(" ", manifest_dir / "08_repaired_sample_image_features.csv")
    print("Summary:")
    print(" ", remote_summary)


if __name__ == "__main__":
    main()
