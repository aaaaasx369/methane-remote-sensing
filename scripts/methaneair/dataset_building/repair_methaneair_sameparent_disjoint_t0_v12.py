#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

HOME = Path.home()
PROJECT = HOME / "methane_release_project"
MF = HOME / "MethaneFuse"

DEFAULT_PAIRED72 = MF / "data/custom/methaneair_sameparent_paired_72h_eval.csv"
DEFAULT_NEG_CANON = PROJECT / "MethaneAIR_Validated_S2_Controls_368_v1/canonical/00_canonical_368_controls.csv"
DEFAULT_NEG_EVAL = MF / "data/custom/methaneair_validated_368_strictqa_eval.csv"
DEFAULT_OUT72 = MF / "data/custom/methaneair_sameparent_paired_72h_disjoint_t0_eval.csv"
DEFAULT_OUT24 = MF / "data/custom/methaneair_sameparent_paired_24h_disjoint_t0_eval.csv"
DEFAULT_AUDIT = PROJECT / "MethaneAIR_S2_SameParent_Paired_Benchmark_v1/manifests/08_disjoint_t0_repair_audit.csv"
DEFAULT_SUMMARY = PROJECT / "MethaneAIR_S2_SameParent_Paired_Benchmark_v1/manifests/09_disjoint_t0_repair_summary.txt"

OVERPASS_MINUTES = 20.0
STRICT24_HOURS = 24.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Repair same-parent paired benchmark by enforcing a distinct Sentinel-2 t0 observation between positive and negative rows. No Earth Engine queries/downloads."
    )
    p.add_argument("--paired72", default=str(DEFAULT_PAIRED72))
    p.add_argument("--negative-canonical", default=str(DEFAULT_NEG_CANON))
    p.add_argument("--negative-eval", default=str(DEFAULT_NEG_EVAL))
    p.add_argument("--out72", default=str(DEFAULT_OUT72))
    p.add_argument("--out24", default=str(DEFAULT_OUT24))
    p.add_argument("--audit", default=str(DEFAULT_AUDIT))
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    p.add_argument("--overpass-minutes", type=float, default=OVERPASS_MINUTES)
    p.add_argument("--allow-missing-t0-time", action="store_true")
    return p.parse_args()


def text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_time(x: Any):
    return pd.to_datetime(x, utc=True, errors="coerce")


def resolve_path(value: Any) -> Path | None:
    s = text(value)
    if not s or s.lower() in {"nan", "none"}:
        return None
    p = Path(s).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates += [MF / p, PROJECT / p, HOME / p]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


_fp_cache: dict[str, str | None] = {}


def raster_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    key = str(path)
    if key in _fp_cache:
        return _fp_cache[key]
    try:
        with rasterio.open(path) as src:
            arr = src.read()
            h = hashlib.sha256()
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(str(src.crs).encode())
            h.update(str(tuple(src.transform)).encode())
            h.update(arr.tobytes(order="C"))
            digest = h.hexdigest()
    except Exception:
        digest = None
    _fp_cache[key] = digest
    return digest


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def grade_priority(v: Any) -> int:
    return {
        "B1_STRONG_HIGH_RES_NO_L4_DETECTION": 0,
        "B2_HIGH_RES_NO_L4_DETECTION_BACKGROUND_WEAK": 1,
    }.get(text(v), 9)


def load_candidates(canon_path: Path, neg_eval_path: Path) -> pd.DataFrame:
    canon = pd.read_csv(canon_path, low_memory=False)
    neg = pd.read_csv(neg_eval_path, low_memory=False)

    require_columns(canon, ["control_id", "Source Positive Record ID", "Final Evidence Grade"], "negative canonical")
    require_columns(neg, ["id", "s2_0_path", "s2_90_path", "s2_360_path"], "negative eval")

    rename = {}
    for c in neg.columns:
        if c != "id" and c in canon.columns:
            rename[c] = f"{c}__eval"
    neg = neg.rename(columns=rename)

    m = canon.merge(
        neg,
        left_on="control_id",
        right_on="id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "__eval"),
    )

    for col in ["s2_0_path", "s2_90_path", "s2_360_path"]:
        if col not in m.columns and f"{col}__eval" in m.columns:
            m[col] = m[f"{col}__eval"]
        if col not in m.columns:
            raise RuntimeError(f"Joined negative table missing {col}")

    if "scene_id" not in m.columns:
        if "scene_id__eval" in m.columns:
            m["scene_id"] = m["scene_id__eval"]
        elif "S2 Product ID" in m.columns:
            m["scene_id"] = m["S2 Product ID"]
        else:
            m["scene_id"] = ""

    if "acquisition_time_utc" not in m.columns:
        if "acquisition_time_utc__eval" in m.columns:
            m["acquisition_time_utc"] = m["acquisition_time_utc__eval"]
        elif "S2 Datetime UTC" in m.columns:
            m["acquisition_time_utc"] = m["S2 Datetime UTC"]
        else:
            m["acquisition_time_utc"] = ""

    m["_grade_priority"] = m["Final Evidence Grade"].map(grade_priority)
    m["_delta"] = pd.to_numeric(
        m["Minimum Absolute S2 Delta Hours"] if "Minimum Absolute S2 Delta Hours" in m.columns else np.nan,
        errors="coerce",
    ).fillna(1e9)
    m["_clear"] = pd.to_numeric(
        m["S2 Clear Over Requested Fraction"] if "S2 Clear Over Requested Fraction" in m.columns else np.nan,
        errors="coerce",
    ).fillna(-1)
    m["_support"] = pd.to_numeric(
        m["Supporting MethaneAIR Flight Count"] if "Supporting MethaneAIR Flight Count" in m.columns else np.nan,
        errors="coerce",
    ).fillna(0)

    return m.sort_values(
        ["Source Positive Record ID", "_grade_priority", "_delta", "_clear", "_support", "control_id"],
        ascending=[True, True, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def candidate_time(c: pd.Series):
    for col in ["acquisition_time_utc", "S2 Datetime UTC", "acquisition_time_utc__eval"]:
        if col in c.index:
            t = parse_time(c[col])
            if not pd.isna(t):
                return t
    return pd.NaT


def candidate_product(c: pd.Series) -> str:
    for col in ["S2 Product ID", "scene_id", "scene_id__eval"]:
        if col in c.index:
            s = text(c[col])
            if s:
                return s
    return ""


def build_negative_row(template: pd.Series, candidate: pd.Series, positive: pd.Series, pair_id: str) -> pd.Series:
    out = template.copy()
    cid = text(candidate["control_id"])
    grade = text(candidate["Final Evidence Grade"])

    out["pair_id"] = pair_id
    out["pair_role"] = "negative"
    out["label"] = 0
    out["id"] = cid
    if "sample_id" in out.index:
        out["sample_id"] = cid

    for c in [
        "site", "source_positive_record_id", "parent_positive_date",
        "parent_positive_datetime_utc", "lat", "lon",
        "positive_t0_abs_delta_hours",
    ]:
        if c in out.index and c in positive.index:
            out[c] = positive[c]

    if "negative_evidence_grade" in out.index:
        out["negative_evidence_grade"] = grade
    if "label_provenance" in out.index:
        out["label_provenance"] = grade
    if "ground_truth_type" in out.index:
        out["ground_truth_type"] = "high_res_no_L4_detection_temporal_control"

    for c in ["s2_0_path", "s2_90_path", "s2_360_path"]:
        out[c] = candidate[c]

    if "scene_id" in out.index:
        out["scene_id"] = candidate_product(candidate)
    if "acquisition_time_utc" in out.index:
        t = candidate_time(candidate)
        out["acquisition_time_utc"] = "" if pd.isna(t) else t.isoformat()

    return out


def main():
    args = parse_args()

    paired_path = Path(args.paired72).expanduser()
    canon_path = Path(args.negative_canonical).expanduser()
    neg_eval_path = Path(args.negative_eval).expanduser()
    out72 = Path(args.out72).expanduser()
    out24 = Path(args.out24).expanduser()
    audit_path = Path(args.audit).expanduser()
    summary_path = Path(args.summary).expanduser()

    for p in [paired_path, canon_path, neg_eval_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    paired = pd.read_csv(paired_path, low_memory=False)
    require_columns(
        paired,
        ["pair_id", "pair_role", "id", "source_positive_record_id",
         "acquisition_time_utc", "s2_0_path", "s2_90_path", "s2_360_path",
         "positive_t0_abs_delta_hours"],
        "paired72 eval",
    )

    candidates = load_candidates(canon_path, neg_eval_path)
    output_rows = []
    audit_rows = []

    pair_ids = list(dict.fromkeys(paired["pair_id"].astype(str)))

    for pair_id in pair_ids:
        g = paired[paired["pair_id"].astype(str).eq(pair_id)]
        pos_g = g[g["pair_role"].astype(str).str.lower().eq("positive")]
        neg_g = g[g["pair_role"].astype(str).str.lower().eq("negative")]
        if len(pos_g) != 1 or len(neg_g) != 1:
            raise RuntimeError(f"Pair structure error {pair_id}: positive={len(pos_g)} negative={len(neg_g)}")

        pos = pos_g.iloc[0]
        current_neg = neg_g.iloc[0]

        source_id = text(pos["source_positive_record_id"])
        pos_path = resolve_path(pos["s2_0_path"])
        pos_fp = raster_fingerprint(pos_path)
        pos_time = parse_time(pos["acquisition_time_utc"])
        pos_product = text(pos["scene_id"]) if "scene_id" in pos.index else ""

        if pos_fp is None:
            raise RuntimeError(f"Unreadable positive t0 for {pair_id}: {pos['s2_0_path']}")
        if pd.isna(pos_time):
            raise RuntimeError(f"Missing positive t0 acquisition time for {pair_id}")

        parent_candidates = candidates[
            candidates["Source Positive Record ID"].astype(str).eq(source_id)
        ].copy()

        valid = []
        for _, c in parent_candidates.iterrows():
            c_path = resolve_path(c["s2_0_path"])
            c_fp = raster_fingerprint(c_path)
            c_time = candidate_time(c)
            c_product = candidate_product(c)

            same_raster = bool(c_fp is not None and c_fp == pos_fp)
            same_product = bool(c_product and pos_product and c_product == pos_product)

            if pd.isna(c_time):
                diff_min = np.nan
                time_ok = bool(args.allow_missing_t0_time)
            else:
                diff_min = abs((c_time - pos_time).total_seconds()) / 60.0
                time_ok = diff_min > float(args.overpass_minutes) + 1e-9

            eligible = bool(c_fp is not None and not same_raster and not same_product and time_ok)
            if eligible:
                valid.append((c, diff_min))

        current_id = text(current_neg["id"])

        if valid:
            chosen, diff_min = valid[0]
            chosen_id = text(chosen["control_id"])
            grade = text(chosen["Final Evidence Grade"])
            status = "KEEP_CURRENT_DISTINCT_T0" if chosen_id == current_id else "REPLACED_WITH_ALTERNATE_DISTINCT_T0"

            pos_out = pos.copy()
            if "negative_evidence_grade" in pos_out.index:
                pos_out["negative_evidence_grade"] = grade
            neg_out = build_negative_row(current_neg, chosen, pos, pair_id)

            output_rows.extend([pos_out, neg_out])
            audit_rows.append({
                "pair_id": pair_id,
                "source_positive_record_id": source_id,
                "site": text(pos["site"]) if "site" in pos.index else "",
                "current_negative_id": current_id,
                "selected_negative_id": chosen_id,
                "selection_status": status,
                "candidate_controls_for_parent": len(parent_candidates),
                "eligible_distinct_t0_controls": len(valid),
                "selected_grade": grade,
                "positive_t0_product": pos_product,
                "selected_negative_t0_product": candidate_product(chosen),
                "selected_t0_time_difference_minutes": diff_min,
            })
        else:
            audit_rows.append({
                "pair_id": pair_id,
                "source_positive_record_id": source_id,
                "site": text(pos["site"]) if "site" in pos.index else "",
                "current_negative_id": current_id,
                "selected_negative_id": "",
                "selection_status": "DROP_NO_DISTINCT_T0_CONTROL",
                "candidate_controls_for_parent": len(parent_candidates),
                "eligible_distinct_t0_controls": 0,
                "selected_grade": "",
                "positive_t0_product": pos_product,
                "selected_negative_t0_product": "",
                "selected_t0_time_difference_minutes": np.nan,
            })

    repaired = pd.DataFrame(output_rows)
    audit = pd.DataFrame(audit_rows)

    if len(repaired):
        repaired = repaired[paired.columns].copy()
        repaired = repaired.sort_values(["pair_id", "label"], ascending=[True, False], kind="mergesort").reset_index(drop=True)

    # Post-repair hard validation: no same t0 raster and no same <=20 min overpass.
    post_rows = []
    for pair_id, g in repaired.groupby("pair_id", sort=False):
        pos = g[g["pair_role"].astype(str).str.lower().eq("positive")].iloc[0]
        neg = g[g["pair_role"].astype(str).str.lower().eq("negative")].iloc[0]

        pfp = raster_fingerprint(resolve_path(pos["s2_0_path"]))
        nfp = raster_fingerprint(resolve_path(neg["s2_0_path"]))
        pt = parse_time(pos["acquisition_time_utc"])
        nt = parse_time(neg["acquisition_time_utc"])

        same_raster = bool(pfp is not None and nfp is not None and pfp == nfp)
        diff_min = abs((nt - pt).total_seconds()) / 60.0 if not pd.isna(pt) and not pd.isna(nt) else np.nan
        same_overpass = bool(diff_min <= float(args.overpass_minutes) + 1e-9) if np.isfinite(diff_min) else None

        post_rows.append({
            "pair_id": pair_id,
            "post_same_t0_raster": same_raster,
            "post_same_t0_overpass": same_overpass,
            "post_t0_diff_minutes": diff_min,
        })

    post = pd.DataFrame(post_rows)
    audit = audit.merge(post, on="pair_id", how="left")

    same_raster_n = int(audit["post_same_t0_raster"].fillna(False).sum())
    same_overpass_n = int(audit["post_same_t0_overpass"].fillna(False).sum())
    if same_raster_n or same_overpass_n:
        raise RuntimeError(
            f"Post-repair validation failed: same_t0_raster={same_raster_n}, same_t0_overpass={same_overpass_n}"
        )

    out72.parent.mkdir(parents=True, exist_ok=True)
    out24.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    repaired.to_csv(out72, index=False)

    if len(repaired):
        positive = repaired[repaired["pair_role"].astype(str).str.lower().eq("positive")].copy()
        positive["_d"] = pd.to_numeric(positive["positive_t0_abs_delta_hours"], errors="coerce")
        strict24_ids = set(positive.loc[positive["_d"].le(STRICT24_HOURS + 1e-9), "pair_id"].astype(str))
        repaired24 = repaired[repaired["pair_id"].astype(str).isin(strict24_ids)].copy()
    else:
        repaired24 = repaired.copy()

    repaired24.to_csv(out24, index=False)
    audit.to_csv(audit_path, index=False)

    final_neg = repaired[repaired["pair_role"].astype(str).str.lower().eq("negative")] if len(repaired) else pd.DataFrame()
    final_pos = repaired[repaired["pair_role"].astype(str).str.lower().eq("positive")] if len(repaired) else pd.DataFrame()

    lines = [
        "METHANEAIR SAME-PARENT DISJOINT-T0 REPAIR V12",
        "=" * 72,
        f"Input ready pairs                  : {len(pair_ids)}",
        f"Final disjoint-t0 72h pairs        : {repaired['pair_id'].nunique() if len(repaired) else 0}",
        f"Final disjoint-t0 72h eval rows    : {len(repaired)}",
        f"Final disjoint-t0 24h pairs        : {repaired24['pair_id'].nunique() if len(repaired24) else 0}",
        f"Final disjoint-t0 24h eval rows    : {len(repaired24)}",
        f"Post-repair same t0 raster         : {same_raster_n}",
        f"Post-repair same <=20m overpass    : {same_overpass_n}",
        "",
        "SELECTION STATUS",
        audit["selection_status"].value_counts(dropna=False).to_string(),
        "",
        "FINAL NEGATIVE EVIDENCE GRADE",
        final_neg["negative_evidence_grade"].value_counts(dropna=False).to_string() if len(final_neg) else "(none)",
        "",
        "FINAL 72H PAIRS BY SITE",
        final_pos["site"].value_counts(dropna=False).to_string() if len(final_pos) else "(none)",
        "",
        "OUTPUTS",
        str(out72),
        str(out24),
        str(audit_path),
    ]
    summary = "\n".join(lines)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\nSummary:", summary_path)


if __name__ == "__main__":
    main()
