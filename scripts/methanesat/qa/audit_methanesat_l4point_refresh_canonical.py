#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_methanesat_l4point_refresh_canonical.py

Canonicalize the live MethaneSAT L4 refresh before treating downloaded crops as
independent positive samples.

Why this is needed
------------------
The live L4 catalogue can contain multiple processing IDs for the same physical
detection. The first refresh script intentionally erred on the side of not
silently merging them. This audit:

1) uses the old inventory's numeric plume_id as the likely scene-local plume ID
   and compares it to live plume_id_in_scene;
2) identifies exact duplicate live detections across processing IDs;
3) flags close cross-processing pairs for review rather than auto-merging them;
4) joins the download audit to determine how many valid, conservatively unique
   new positive TIFFs actually exist.

It NEVER deletes TIFFs.

Inputs expected in the refresh output directory:
  01_old_positive_inventory.csv
  02_l4point_diff.csv
  06_new_positive_download_audit.csv

Outputs:
  10_revised_l4point_diff.csv
  11_exact_processing_duplicate_groups.csv
  12_possible_processing_duplicates_review.csv
  13_valid_new_positive_unique_conservative.csv
  14_valid_new_positive_all_clear.csv
  SUMMARY_CANONICAL_REFRESH_AUDIT.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--refresh-dir",
        default="/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/04_l4point_refresh",
    )
    p.add_argument(
        "--exact-duplicate-distance-m",
        type=float,
        default=20.0,
        help="Same collection and <= this distance can be exact-duplicate candidates.",
    )
    p.add_argument(
        "--same-scene-id-distance-m",
        type=float,
        default=1500.0,
        help="Same collection + same scene-local plume ID + <= this distance => old revised match.",
    )
    p.add_argument(
        "--possible-duplicate-distance-m",
        type=float,
        default=1000.0,
        help="Same collection, different processing IDs, and <= this distance => review pair.",
    )
    return p.parse_args()


def norm_collection(v):
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def norm_scene_plume_id(v):
    """Normalize scene-local numeric IDs such as 6.0 -> '6'."""
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none", "null", ""}:
        return ""
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return str(int(float(s)))
    return s


def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371008.8
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rel_diff(a, b):
    try:
        a, b = float(a), float(b)
    except Exception:
        return np.nan
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    args = parse_args()
    root = Path(args.refresh_dir).expanduser()

    old_path = root / "01_old_positive_inventory.csv"
    diff_path = root / "02_l4point_diff.csv"
    audit_path = root / "06_new_positive_download_audit.csv"

    for p in [old_path, diff_path, audit_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    old = pd.read_csv(old_path, low_memory=False)
    live = pd.read_csv(diff_path, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)

    old["collection_norm"] = old["collection_id"].map(norm_collection)
    old["scene_plume_id_norm"] = old["plume_id"].map(norm_scene_plume_id)

    live["collection_norm"] = live["collection_id"].map(norm_collection)
    live["scene_plume_id_norm"] = live["plume_id_in_scene"].map(norm_scene_plume_id)

    # ------------------------------------------------------------------
    # 1. Revised old-vs-live classification using scene-local plume ID.
    # ------------------------------------------------------------------
    revised_rows = []

    old_by_cid = {cid: g.copy() for cid, g in old.groupby("collection_norm")}

    for _, r in live.iterrows():
        rec = r.to_dict()
        g = old_by_cid.get(r["collection_norm"], pd.DataFrame())

        rec["scene_id_old_match_count"] = 0
        rec["scene_id_nearest_old_m"] = np.nan
        rec["scene_id_nearest_old_sample"] = ""
        rec["revised_match_status"] = r["old_match_status"]
        rec["revised_match_reason"] = r["old_match_method"]

        if not g.empty and r["scene_plume_id_norm"]:
            same_id = g[
                g["scene_plume_id_norm"].eq(r["scene_plume_id_norm"])
            ].copy()

            rec["scene_id_old_match_count"] = int(len(same_id))

            if not same_id.empty:
                dists = []
                for _, o in same_id.iterrows():
                    dists.append(
                        haversine_m(
                            r["latitude"], r["longitude"],
                            o["latitude"], o["longitude"]
                        )
                    )
                same_id["_distance_m"] = dists
                same_id = same_id[same_id["_distance_m"].notna()]

                if not same_id.empty:
                    j = same_id["_distance_m"].idxmin()
                    d = float(same_id.loc[j, "_distance_m"])
                    rec["scene_id_nearest_old_m"] = d
                    rec["scene_id_nearest_old_sample"] = str(
                        same_id.loc[j, "old_sample_id"]
                    )

                    if d <= args.same_scene_id_distance_m:
                        rec["revised_match_status"] = "ALREADY_HAVE_REVISED_SAME_SCENE_ID"
                        rec["revised_match_reason"] = (
                            "same_collection_same_scene_local_plume_id_"
                            f"within_{args.same_scene_id_distance_m:.0f}m"
                        )

        revised_rows.append(rec)

    revised = pd.DataFrame(revised_rows)
    revised.to_csv(root / "10_revised_l4point_diff.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Exact duplicate detection among live rows.
    #    Exact = same collection + across processing IDs + <=20m +
    #            same scene-local plume ID OR nearly identical flux.
    # ------------------------------------------------------------------
    n = len(revised)
    uf = UnionFind(n)
    exact_pairs = []
    possible_pairs = []

    for cid, g in revised.groupby("collection_norm"):
        idxs = list(g.index)
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                i, j = idxs[ai], idxs[bi]
                a, b = revised.loc[i], revised.loc[j]

                if pd.isna(a.get("processing_id")) or pd.isna(b.get("processing_id")):
                    continue
                if str(a.get("processing_id")) == str(b.get("processing_id")):
                    continue

                d = haversine_m(
                    a["latitude"], a["longitude"],
                    b["latitude"], b["longitude"]
                )
                if not np.isfinite(d):
                    continue

                same_scene_id = (
                    a["scene_plume_id_norm"] != ""
                    and a["scene_plume_id_norm"] == b["scene_plume_id_norm"]
                )
                flux_rd = rel_diff(a.get("flux"), b.get("flux"))

                if (
                    d <= args.exact_duplicate_distance_m
                    and (same_scene_id or (np.isfinite(flux_rd) and flux_rd <= 0.01))
                ):
                    uf.union(i, j)
                    exact_pairs.append({
                        "collection_id": cid,
                        "row_i": i,
                        "row_j": j,
                        "plume_id_i": a.get("plume_id"),
                        "plume_id_j": b.get("plume_id"),
                        "processing_id_i": a.get("processing_id"),
                        "processing_id_j": b.get("processing_id"),
                        "scene_plume_id_i": a.get("plume_id_in_scene"),
                        "scene_plume_id_j": b.get("plume_id_in_scene"),
                        "distance_m": d,
                        "flux_i": a.get("flux"),
                        "flux_j": b.get("flux"),
                        "flux_relative_difference": flux_rd,
                    })
                elif d <= args.possible_duplicate_distance_m:
                    possible_pairs.append({
                        "collection_id": cid,
                        "row_i": i,
                        "row_j": j,
                        "plume_id_i": a.get("plume_id"),
                        "plume_id_j": b.get("plume_id"),
                        "processing_id_i": a.get("processing_id"),
                        "processing_id_j": b.get("processing_id"),
                        "scene_plume_id_i": a.get("plume_id_in_scene"),
                        "scene_plume_id_j": b.get("plume_id_in_scene"),
                        "distance_m": d,
                        "flux_i": a.get("flux"),
                        "flux_j": b.get("flux"),
                        "flux_relative_difference": flux_rd,
                        "same_scene_local_plume_id": same_scene_id,
                    })

    # Build duplicate group table.
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    duplicate_groups = []
    group_id_for_index = {}
    gid = 0
    for inds in groups.values():
        if len(inds) <= 1:
            continue
        gid += 1
        for pos, idx in enumerate(inds):
            r = revised.loc[idx]
            group_id_for_index[idx] = gid
            duplicate_groups.append({
                "duplicate_group_id": gid,
                "canonical_keep": pos == 0,
                "live_index": r.get("live_index"),
                "collection_id": r.get("collection_id"),
                "plume_id": r.get("plume_id"),
                "plume_id_in_scene": r.get("plume_id_in_scene"),
                "processing_id": r.get("processing_id"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "flux": r.get("flux"),
                "old_match_status": r.get("old_match_status"),
                "revised_match_status": r.get("revised_match_status"),
            })

    dup_df = pd.DataFrame(duplicate_groups)
    dup_df.to_csv(root / "11_exact_processing_duplicate_groups.csv", index=False)

    possible_df = pd.DataFrame(possible_pairs)
    possible_df.to_csv(
        root / "12_possible_processing_duplicates_review.csv",
        index=False
    )

    revised["exact_duplicate_group_id"] = [
        group_id_for_index.get(i, np.nan) for i in range(n)
    ]

    # Within each exact duplicate group, keep one row only.
    keep_index = set(range(n))
    for inds in groups.values():
        if len(inds) > 1:
            for idx in inds[1:]:
                keep_index.discard(idx)

    revised["canonical_keep_after_exact_dedup"] = [
        i in keep_index for i in range(n)
    ]
    revised.to_csv(root / "10_revised_l4point_diff.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Join with download audit to count VALID new positive TIFFs.
    # ------------------------------------------------------------------
    # Normalize boolean field robustly.
    if "final_valid" not in audit.columns:
        raise ValueError("06_new_positive_download_audit.csv has no final_valid column")
    audit["_final_valid_bool"] = (
        audit["final_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    # Use plume_id as the strongest join key generated by refresh script.
    live_key_cols = ["collection_id", "plume_id"]
    a = audit.copy()
    a["collection_norm"] = a["collection_id"].map(norm_collection)
    a["plume_id_norm"] = a["plume_id"].astype(str).str.strip()

    rv = revised.copy()
    rv["plume_id_norm"] = rv["plume_id"].astype(str).str.strip()

    valid_clear = a[
        a["_final_valid_bool"]
        & a["old_match_status"].eq("CLEAR_NEW_FEATURE")
    ].copy()

    # Attach exact-duplicate canonical status from revised table.
    map_cols = rv[
        [
            "collection_norm",
            "plume_id_norm",
            "canonical_keep_after_exact_dedup",
            "exact_duplicate_group_id",
            "revised_match_status",
        ]
    ].drop_duplicates(["collection_norm", "plume_id_norm"])

    valid_clear = valid_clear.merge(
        map_cols,
        on=["collection_norm", "plume_id_norm"],
        how="left",
    )

    valid_clear.to_csv(
        root / "14_valid_new_positive_all_clear.csv",
        index=False,
    )

    unique_conservative = valid_clear[
        valid_clear["canonical_keep_after_exact_dedup"].fillna(True)
    ].copy()

    # A row reclassified as already-have by same scene-local plume ID should
    # not inflate the "new" count.
    unique_conservative = unique_conservative[
        ~unique_conservative["revised_match_status"].eq(
            "ALREADY_HAVE_REVISED_SAME_SCENE_ID"
        )
    ].copy()

    unique_conservative.to_csv(
        root / "13_valid_new_positive_unique_conservative.csv",
        index=False,
    )

    failed = audit[~audit["_final_valid_bool"]].copy()

    counts_old = revised["old_match_status"].value_counts().to_dict()
    counts_revised = revised["revised_match_status"].value_counts().to_dict()

    lines = [
        "# MethaneSAT L4 refresh canonical audit",
        "",
        "## Input",
        f"- Live L4 rows: {len(revised)}",
        f"- Old positive rows: {len(old)}",
        f"- Download audit rows: {len(audit)}",
        "",
        "## Original diff",
    ]
    for k in sorted(counts_old):
        lines.append(f"- {k}: {counts_old[k]}")

    lines += ["", "## Revised diff using scene-local plume ID"]
    for k in sorted(counts_revised):
        lines.append(f"- {k}: {counts_revised[k]}")

    lines += [
        "",
        "## Processing-version duplication",
        f"- Exact duplicate groups: {len(dup_df['duplicate_group_id'].unique()) if len(dup_df) else 0}",
        f"- Rows participating in exact duplicate groups: {len(dup_df)}",
        f"- Possible close cross-processing pairs requiring review: {len(possible_df)}",
        "",
        "## Downloaded new positives",
        f"- Valid clear-new TIFF rows before canonicalization: {len(valid_clear)}",
        f"- Valid conservatively unique new positive TIFFs: {len(unique_conservative)}",
        f"- Missing/invalid download rows: {len(failed)}",
        "",
        "## Important interpretation",
        "- Do not report 48 downloaded TIFFs as 48 independent physical positive detections until canonicalization.",
        "- Exact processing duplicates are merged conservatively.",
        "- Close cross-processing pairs are only flagged; they are NOT auto-merged.",
        "- 13_valid_new_positive_unique_conservative.csv is the safest new-positive set for the next pipeline stage.",
    ]

    (root / "SUMMARY_CANONICAL_REFRESH_AUDIT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print("METHANESAT L4 REFRESH CANONICAL AUDIT")
    print("=" * 80)
    print("Original live rows                    :", len(revised))
    print("Exact processing duplicate groups     :", len(dup_df['duplicate_group_id'].unique()) if len(dup_df) else 0)
    print("Possible close processing pairs       :", len(possible_df))
    print("Valid clear-new TIFF rows             :", len(valid_clear))
    print("Valid conservatively unique new TIFFs :", len(unique_conservative))
    print("Missing/invalid                       :", len(failed))
    print()
    print("Upload:")
    for fn in [
        "SUMMARY_CANONICAL_REFRESH_AUDIT.md",
        "10_revised_l4point_diff.csv",
        "11_exact_processing_duplicate_groups.csv",
        "12_possible_processing_duplicates_review.csv",
        "13_valid_new_positive_unique_conservative.csv",
    ]:
        print(" ", root / fn)


if __name__ == "__main__":
    main()
