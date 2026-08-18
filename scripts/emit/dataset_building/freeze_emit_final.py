#!/usr/bin/env python3
"""
Create the canonical FINAL FREEZE for the EMIT MethaneFuse external benchmark.

Default output:
  ~/methane_release_project/EMIT_MethaneFuse_FINAL_FREEZE

The freeze contains:
  - the final 19-pair / 38-sample real-temporal dataset (114 TIFFs)
  - four portable experiment manifests
  - four canonical MethaneFuse metric JSONs
  - result summary tables and final scientific interpretation
  - temporal search / pixel-QA / pairing provenance where available
  - preprocessing / ablation scripts where available
  - environment + MethaneFuse git/checkpoint fingerprint
  - SHA256 checksums and file inventory

No data are downloaded from the internet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

try:
    import rasterio
except Exception as exc:
    raise SystemExit(
        "rasterio is required for TIFF validation. "
        "Activate the same .venv311 environment used for preprocessing.\n"
        f"Import error: {exc}"
    )

HOME = Path.home()
PROJECT = HOME / "methane_release_project"
MF = HOME / "MethaneFuse"

SRC_TEMPORAL = PROJECT / "EMIT_MethaneFuse_480m_TEMPORAL_60D"
SRC_SMOKE = PROJECT / "EMIT_MethaneFuse_480m_SMOKE_HANDOFF"
SRC_SEARCH = PROJECT / "emit_temporal_search"

DEFAULT_OUT = PROJECT / "EMIT_MethaneFuse_FINAL_FREEZE"

METRIC_SOURCES = {
    "expanded60_real_temporal": MF / "results/eval/emit_temporal60_full.json",
    "expanded60_t0_repeated": MF / "results/eval/emit_temporal60_t0repeated_same38.json",
    "strict30_real_temporal": MF / "results/eval/emit_temporal_strict30_real.json",
    "strict30_t0_repeated": MF / "results/eval/emit_temporal_strict30_t0repeated.json",
}

MANIFEST_SOURCES = {
    "expanded60_real_temporal":
        SRC_TEMPORAL / "eval_emit_480m_abs.csv",
    "expanded60_t0_repeated":
        SRC_TEMPORAL / "eval_emit_480m_abs_t0repeated.csv",
    "strict30_real_temporal":
        SRC_TEMPORAL / "eval_emit_480m_abs_strict30_real.csv",
    "strict30_t0_repeated":
        SRC_TEMPORAL / "eval_emit_480m_abs_strict30_t0repeated.csv",
}

SCRIPT_CANDIDATES = [
    PROJECT / "prepare_emit_for_methanefuse_v4_resume_rflonly.py",
    PROJECT / "build_emit_real_temporal_fixed.py",
    PROJECT / "make_emit_t0_repeated_same38.py",
    PROJECT / "compare_emit_temporal_ablation.py",
    PROJECT / "make_emit_strict30_sensitivity.py",
    PROJECT / "compare_emit_strict30_sensitivity.py",
]

PROVENANCE_CANDIDATES = [
    SRC_TEMPORAL / "temporal_build_progress.csv",
    SRC_TEMPORAL / "eval_emit_480m_temporal_all_pass.csv",
    SRC_TEMPORAL / "strict30_pair_audit.csv",
    SRC_TEMPORAL / "WV3_VNIR_SWIR_response.csv",
    SRC_TEMPORAL / "README_FOR_SENIOR.md",
    SRC_SEARCH / "temporal_selection.csv",
    SRC_SEARCH / "temporal_pair_coverage.csv",
    SRC_SMOKE / "qa_report.csv",
    SRC_SMOKE / "pair_anchor_audit.csv",
]

EXPECTED_EXPANDED_ROWS = 38
EXPECTED_EXPANDED_PAIRS = 19
EXPECTED_STRICT_ROWS = 22
EXPECTED_STRICT_PAIRS = 11
EXPECTED_TIFS = 114


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_text(cmd: List[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def confusion_from_metrics(n_pos: int, n_neg: int, overall: Dict[str, Any]):
    recall = float(overall["recall"])
    fpr = float(overall["fpr"])
    tp = int(round(recall * n_pos))
    fn = n_pos - tp
    fp = int(round(fpr * n_neg))
    tn = n_neg - fp
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def validate_manifest(
    name: str,
    df: pd.DataFrame,
    expected_rows: int,
    expected_pairs: int,
):
    required = [
        "id", "pair_id", "label",
        "emit_0_path", "emit_90_path", "emit_360_path",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}")

    rows = len(df)
    pairs = df["pair_id"].nunique()
    pos = int((pd.to_numeric(df["label"]) == 1).sum())
    neg = int((pd.to_numeric(df["label"]) == 0).sum())

    if rows != expected_rows:
        raise RuntimeError(f"{name}: rows={rows}, expected={expected_rows}")
    if pairs != expected_pairs:
        raise RuntimeError(f"{name}: pairs={pairs}, expected={expected_pairs}")
    if pos != neg or pos + neg != rows:
        raise RuntimeError(
            f"{name}: unbalanced or invalid labels; pos={pos}, neg={neg}, rows={rows}"
        )

    return {"rows": rows, "pairs": pairs, "positive": pos, "negative": neg}


def source_sample_path_from_value(v: Any, sample_id: str, slot: str) -> Path:
    """
    Resolve an evaluator path to a source TIFF.
    Works for absolute paths and relative paths.
    """
    s = str(v)
    p = Path(s).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()

    p2 = SRC_TEMPORAL / s
    if p2.exists():
        return p2.resolve()

    fallback_name = {
        "emit_0_path": "emit_t0.tif",
        "emit_90_path": "emit_tminus90.tif",
        "emit_360_path": "emit_tminus180.tif",
    }[slot]
    p3 = SRC_TEMPORAL / "samples" / str(sample_id) / fallback_name
    if p3.exists():
        return p3.resolve()

    raise FileNotFoundError(
        f"Cannot resolve {slot} for {sample_id}: {v}"
    )


def portable_manifest(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Path, Path]]:
    """
    Convert all model path columns to freeze-relative paths.
    Returns portable dataframe + source->relative destination mapping.
    """
    out = df.copy()
    copy_map: Dict[Path, Path] = {}

    for idx, row in out.iterrows():
        sid = str(row["id"])
        for col in ["emit_0_path", "emit_90_path", "emit_360_path"]:
            src = source_sample_path_from_value(row[col], sid, col)

            # Preserve semantic filename from the actual referenced source.
            dest_rel = Path("samples") / sid / src.name
            copy_map[src] = dest_rel
            out.at[idx, col] = dest_rel.as_posix()

    return out, copy_map


def validate_tif(path: Path):
    with rasterio.open(path) as ds:
        shape = (ds.count, ds.height, ds.width)
        if shape != (16, 518, 518):
            raise RuntimeError(f"Bad TIFF shape {shape}: {path}")
        if any(str(x) != "uint16" for x in ds.dtypes):
            raise RuntimeError(f"Bad TIFF dtype {ds.dtypes}: {path}")


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Final freeze output directory.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing freeze directory before rebuilding.",
    )
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()

    print("=" * 80)
    print("EMIT METHANEFUSE FINAL FREEZE")
    print("=" * 80)
    print("Output:", out)

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------
    required = list(METRIC_SOURCES.values()) + list(MANIFEST_SOURCES.values())
    required += [
        SRC_TEMPORAL / "WV3_VNIR_SWIR_response.csv",
        MF / "checkpoints/classification/methanefuse_cls_480m.pt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nMISSING REQUIRED FILES:")
        for p in missing:
            print("  ", p)
        raise SystemExit(2)

    if out.exists():
        if not args.overwrite:
            raise SystemExit(
                f"\nFreeze directory already exists:\n{out}\n"
                "Re-run with --overwrite only if you intentionally want to rebuild it."
            )
        shutil.rmtree(out)

    for d in [
        out,
        out / "dataset",
        out / "samples",
        out / "metrics",
        out / "provenance",
        out / "scripts",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load + validate manifests
    # ------------------------------------------------------------------
    source_dfs = {
        k: pd.read_csv(v)
        for k, v in MANIFEST_SOURCES.items()
    }

    counts = {}
    counts["expanded60_real_temporal"] = validate_manifest(
        "expanded60_real_temporal",
        source_dfs["expanded60_real_temporal"],
        EXPECTED_EXPANDED_ROWS,
        EXPECTED_EXPANDED_PAIRS,
    )
    counts["expanded60_t0_repeated"] = validate_manifest(
        "expanded60_t0_repeated",
        source_dfs["expanded60_t0_repeated"],
        EXPECTED_EXPANDED_ROWS,
        EXPECTED_EXPANDED_PAIRS,
    )
    counts["strict30_real_temporal"] = validate_manifest(
        "strict30_real_temporal",
        source_dfs["strict30_real_temporal"],
        EXPECTED_STRICT_ROWS,
        EXPECTED_STRICT_PAIRS,
    )
    counts["strict30_t0_repeated"] = validate_manifest(
        "strict30_t0_repeated",
        source_dfs["strict30_t0_repeated"],
        EXPECTED_STRICT_ROWS,
        EXPECTED_STRICT_PAIRS,
    )

    # Ensure matched ablations have identical IDs / labels.
    for a, b in [
        ("expanded60_real_temporal", "expanded60_t0_repeated"),
        ("strict30_real_temporal", "strict30_t0_repeated"),
    ]:
        left = (
            source_dfs[a][["id", "pair_id", "label"]]
            .sort_values(["pair_id", "id"])
            .reset_index(drop=True)
        )
        right = (
            source_dfs[b][["id", "pair_id", "label"]]
            .sort_values(["pair_id", "id"])
            .reset_index(drop=True)
        )
        if not left.equals(right):
            raise RuntimeError(f"Matched ablation mismatch: {a} vs {b}")

    # ------------------------------------------------------------------
    # Create portable manifests and gather all unique source TIFFs
    # ------------------------------------------------------------------
    all_copy_map: Dict[Path, Path] = {}

    manifest_output_names = {
        "expanded60_real_temporal": "expanded60_real_temporal.csv",
        "expanded60_t0_repeated": "expanded60_t0_repeated.csv",
        "strict30_real_temporal": "strict30_real_temporal.csv",
        "strict30_t0_repeated": "strict30_t0_repeated.csv",
    }

    for key, df in source_dfs.items():
        portable, cmap = portable_manifest(df)

        # Verify t0-repeated semantics where appropriate.
        if "t0_repeated" in key:
            if not (
                (portable["emit_90_path"] == portable["emit_0_path"]).all()
                and (portable["emit_360_path"] == portable["emit_0_path"]).all()
            ):
                raise RuntimeError(f"{key}: repeated-t0 semantics are not exact.")

        dest = out / "dataset" / manifest_output_names[key]
        portable.to_csv(dest, index=False)
        all_copy_map.update(cmap)

    # The real expanded dataset should contain 38 * 3 unique physical TIFFs.
    real_portable, real_map = portable_manifest(
        source_dfs["expanded60_real_temporal"]
    )
    unique_real_sources = sorted(real_map.keys(), key=str)
    if len(unique_real_sources) != EXPECTED_TIFS:
        raise RuntimeError(
            f"Expanded real-temporal physical TIFF count is {len(unique_real_sources)}, "
            f"expected {EXPECTED_TIFS}."
        )

    print(f"\nValidated canonical dataset: {EXPECTED_EXPANDED_PAIRS} pairs / "
          f"{EXPECTED_EXPANDED_ROWS} rows / {EXPECTED_TIFS} TIFFs")

    # ------------------------------------------------------------------
    # Copy + validate samples
    # ------------------------------------------------------------------
    print("\nCopying and validating 114 model TIFFs...")
    copied = set()
    for i, src in enumerate(unique_real_sources, 1):
        validate_tif(src)
        rel = real_map[src]
        dst = out / rel
        copy_file(src, dst)
        validate_tif(dst)
        copied.add(dst.resolve())
        if i % 10 == 0 or i == EXPECTED_TIFS:
            print(f"  {i:3d}/{EXPECTED_TIFS}")

    actual_tifs = list((out / "samples").rglob("*.tif"))
    if len(actual_tifs) != EXPECTED_TIFS:
        raise RuntimeError(
            f"Freeze has {len(actual_tifs)} TIFFs, expected {EXPECTED_TIFS}"
        )

    # ------------------------------------------------------------------
    # Copy metric JSONs + extract canonical summary
    # ------------------------------------------------------------------
    metric_data = {}
    results_rows = []

    for key, src in METRIC_SOURCES.items():
        d = load_json(src)
        metric_data[key] = d
        copy_file(src, out / "metrics" / f"{key}.json")

        c = counts[key]
        overall = d["overall"]
        if int(d.get("count", overall.get("count", -1))) != c["rows"]:
            raise RuntimeError(
                f"{key}: metric count does not match manifest rows"
            )

        cm = confusion_from_metrics(
            c["positive"], c["negative"], overall
        )

        results_rows.append({
            "experiment": key,
            "pairs": c["pairs"],
            "samples": c["rows"],
            "positive": c["positive"],
            "negative": c["negative"],
            "loss": float(d["loss"]),
            "accuracy": float(overall["acc"]),
            "auroc": float(overall["auroc"]),
            "recall": float(overall["recall"]),
            "fpr": float(overall["fpr"]),
            **cm,
        })

    write_csv(out / "RESULTS_TABLE.csv", results_rows)

    # ------------------------------------------------------------------
    # Canonical deltas
    # ------------------------------------------------------------------
    e_real = metric_data["expanded60_real_temporal"]
    e_rep = metric_data["expanded60_t0_repeated"]
    s_real = metric_data["strict30_real_temporal"]
    s_rep = metric_data["strict30_t0_repeated"]

    def delta(real, rep):
        ro, bo = real["overall"], rep["overall"]
        return {
            "accuracy": float(ro["acc"]) - float(bo["acc"]),
            "auroc": float(ro["auroc"]) - float(bo["auroc"]),
            "recall": float(ro["recall"]) - float(bo["recall"]),
            "fpr": float(ro["fpr"]) - float(bo["fpr"]),
            "loss": float(real["loss"]) - float(rep["loss"]),
        }

    final_metrics = {
        "benchmark_name": "EMIT MethaneFuse External Real-Temporal Benchmark",
        "checkpoint": "checkpoints/classification/methanefuse_cls_480m.pt",
        "dataset_definition": {
            "expanded60": {
                "pairs": EXPECTED_EXPANDED_PAIRS,
                "samples": EXPECTED_EXPANDED_ROWS,
                "positive": 19,
                "negative": 19,
                "temporal_frames":
                    "real t0 + real acquisition near t-90 + real acquisition near t-180",
                "target_date_tolerance_days": 60,
            },
            "strict30_sensitivity": {
                "pairs": EXPECTED_STRICT_PAIRS,
                "samples": EXPECTED_STRICT_ROWS,
                "positive": 11,
                "negative": 11,
                "target_date_tolerance_days": 30,
            },
        },
        "experiments": {
            k: {
                "count": int(v["count"]),
                "loss": float(v["loss"]),
                "overall": {
                    "acc": float(v["overall"]["acc"]),
                    "auroc": float(v["overall"]["auroc"]),
                    "recall": float(v["overall"]["recall"]),
                    "fpr": float(v["overall"]["fpr"]),
                },
            }
            for k, v in metric_data.items()
        },
        "matched_ablation_delta_real_minus_t0_repeated": {
            "expanded60": delta(e_real, e_rep),
            "strict30": delta(s_real, s_rep),
        },
        "canonical_conclusion":
            "In this EMIT external benchmark, real historical temporal context "
            "did not improve AUROC. This direction persisted in both the expanded "
            "±60-day and strict ±30-day matched ablations.",
        "limitations": [
            "Small benchmark: 19 pairs in expanded60 and 11 pairs in strict30.",
            "Negative labels are same-site different-time candidate no-detection "
            "observations, not independently confirmed zero-emission controls.",
            "EMIT is adapted to the released MethaneFuse interface through "
            "EMIT L2A reflectance -> simulated 16-band WorldView-3 representation.",
            "The current generic MethaneFuse loader column emit_360_path contains "
            "the actual EMIT t-180 frame for compatibility with the loader naming.",
            "Observed differences are descriptive; no claim of statistical "
            "significance is made from this small benchmark.",
        ],
    }

    with (out / "FINAL_METRICS.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Provenance + scripts
    # ------------------------------------------------------------------
    for src in PROVENANCE_CANDIDATES:
        if src.exists():
            copy_file(src, out / "provenance" / src.name)

    for src in SCRIPT_CANDIDATES:
        if src.exists():
            copy_file(src, out / "scripts" / src.name)

    # ------------------------------------------------------------------
    # Environment / fingerprints
    # ------------------------------------------------------------------
    ckpt = MF / "checkpoints/classification/methanefuse_cls_480m.pt"
    checkpoint_sha = sha256_file(ckpt)

    env_text = f"""EMIT MethaneFuse FINAL FREEZE environment
================================================

Created with:
Python: {sys.version.replace(os.linesep, " ")}
Platform: {platform.platform()}

MethaneFuse repo:
Path: {MF}
Git HEAD: {run_text(["git", "rev-parse", "HEAD"], cwd=MF)}
Git status --short:
{run_text(["git", "status", "--short"], cwd=MF)}

Checkpoint:
Path: {ckpt}
SHA256: {checkpoint_sha}

Important interface semantics:
- emit_0_path   = actual t0
- emit_90_path  = actual ~t-90
- emit_360_path = actual ~t-180 for EMIT compatibility with the generic loader
"""
    (out / "ENVIRONMENT.txt").write_text(env_text, encoding="utf-8")

    # ------------------------------------------------------------------
    # README FINAL
    # ------------------------------------------------------------------
    eo = e_real["overall"]
    eb = e_rep["overall"]
    so = s_real["overall"]
    sb = s_rep["overall"]
    ed = delta(e_real, e_rep)
    sd = delta(s_real, s_rep)

    readme = f"""# EMIT MethaneFuse External Benchmark — FINAL FREEZE

## Status

**FREEZE / COMPLETE**

This folder is the canonical frozen endpoint for the EMIT external MethaneFuse
evaluation.

## Canonical dataset

Primary real-temporal benchmark:

- 19 complete matched POS/NEG pairs
- 38 samples total
- 19 positive / 19 candidate negative
- 114 physical TIFFs
- each sample contains:
  - real `t0`
  - real EMIT acquisition near `t-90`
  - real EMIT acquisition near `t-180`
- temporal target tolerance for the primary set: ±60 days
- all retained frames passed pixel QA
- each model TIFF is 16 bands × 518 × 518, uint16

Positive `t0` samples are anchored to published EMIT CH4PLM plume evidence.

Negative `t0` samples are **same-site different-time candidate negatives**
without a published CH4PLM source-scene detection. They are not independently
confirmed zero-emission observations.

## MethaneFuse compatibility

The released generic MethaneFuse loader expects:

- `emit_0_path`
- `emit_90_path`
- `emit_360_path`

For this EMIT benchmark:

- `emit_0_path` = actual `t0`
- `emit_90_path` = actual `~t-90`
- `emit_360_path` = actual `~t-180`

The third column retains the loader's `360` name only for compatibility.

EMIT L2A surface reflectance was converted to the 16-band simulated
WorldView-3 representation used by the released interface.

## Four canonical experiments

### 1. Expanded60 — real temporal
Pairs: 19  
Samples: 38

- Accuracy: {float(eo["acc"]):.6f}
- AUROC: {float(eo["auroc"]):.6f}
- Recall: {float(eo["recall"]):.6f}
- FPR: {float(eo["fpr"]):.6f}
- Loss: {float(e_real["loss"]):.6f}

### 2. Expanded60 — matched t0-repeated baseline
Same 38 samples and labels; only temporal context is replaced by `t0/t0/t0`.

- Accuracy: {float(eb["acc"]):.6f}
- AUROC: {float(eb["auroc"]):.6f}
- Recall: {float(eb["recall"]):.6f}
- FPR: {float(eb["fpr"]):.6f}
- Loss: {float(e_rep["loss"]):.6f}

Matched real-minus-baseline effects:

- ΔAccuracy: {ed["accuracy"]:+.6f}
- ΔAUROC: {ed["auroc"]:+.6f}
- ΔRecall: {ed["recall"]:+.6f}
- ΔFPR: {ed["fpr"]:+.6f}
- ΔLoss: {ed["loss"]:+.6f}

### 3. Strict30 sensitivity — real temporal
Pairs: 11  
Samples: 22

- Accuracy: {float(so["acc"]):.6f}
- AUROC: {float(so["auroc"]):.6f}
- Recall: {float(so["recall"]):.6f}
- FPR: {float(so["fpr"]):.6f}
- Loss: {float(s_real["loss"]):.6f}

### 4. Strict30 sensitivity — matched t0-repeated baseline
Same 22 strict rows; temporal context replaced by `t0/t0/t0`.

- Accuracy: {float(sb["acc"]):.6f}
- AUROC: {float(sb["auroc"]):.6f}
- Recall: {float(sb["recall"]):.6f}
- FPR: {float(sb["fpr"]):.6f}
- Loss: {float(s_rep["loss"]):.6f}

Matched real-minus-baseline effects:

- ΔAccuracy: {sd["accuracy"]:+.6f}
- ΔAUROC: {sd["auroc"]:+.6f}
- ΔRecall: {sd["recall"]:+.6f}
- ΔFPR: {sd["fpr"]:+.6f}
- ΔLoss: {sd["loss"]:+.6f}

## Final interpretation

For the 19-pair primary real-temporal benchmark, MethaneFuse achieved AUROC
{float(eo["auroc"]):.3f}. On the exact same 38 observations with t0 repeated
across all temporal slots, AUROC was {float(eb["auroc"]):.3f}
(ΔAUROC = {ed["auroc"]:+.3f} for real temporal context).

The same direction persisted under the stricter ±30-day temporal sensitivity:
real-temporal AUROC was {float(so["auroc"]):.3f}, compared with
{float(sb["auroc"]):.3f} for the matched t0-repeated baseline
(ΔAUROC = {sd["auroc"]:+.3f}).

Therefore:

> **In this EMIT external benchmark, real historical temporal context did not
> improve overall AUROC discrimination, and this finding was not explained
> simply by the wider ±60-day temporal matching tolerance.**

At the fixed classification threshold, real historical context did reduce the
false-positive rate in both matched analyses, so the result should not be
described as uniform degradation across every metric.

## Limitations

1. The benchmark is small: 19 pairs in the primary analysis and 11 pairs in
   the strict sensitivity subset.
2. Candidate negatives are no-published-detection controls, not independently
   verified methane-free observations.
3. This is an external-domain EMIT evaluation through the released
   MethaneFuse-compatible simulated WV3 representation.
4. These differences are descriptive; this freeze does not claim statistical
   significance.
5. Results should not be generalized to claim that historical imagery is
   universally harmful. The conclusion is specific to this EMIT benchmark.

## Folder contents

- `dataset/`
  - portable manifests for all four canonical experiments
- `samples/`
  - the 114 frozen real-temporal model TIFFs
- `metrics/`
  - original four MethaneFuse result JSONs
- `FINAL_METRICS.json`
  - canonical machine-readable summary
- `RESULTS_TABLE.csv`
  - one-row-per-experiment result table with confusion counts
- `provenance/`
  - temporal search, pair audit, QA, and SRF records where available
- `scripts/`
  - preprocessing and ablation scripts where available
- `ENVIRONMENT.txt`
  - MethaneFuse git state and checkpoint fingerprint
- `FREEZE_INVENTORY.csv`
  - file sizes and SHA256 digests
- `SHA256SUMS.txt`
  - integrity checksums

## Canonical scientific status

**EMIT analysis: FREEZE / COMPLETE.**

Further temporal-tolerance expansion is not part of the canonical analysis.
"""
    (out / "README_FINAL.md").write_text(readme, encoding="utf-8")

    # ------------------------------------------------------------------
    # Inventory + hashes
    # ------------------------------------------------------------------
    print("\nComputing SHA256 inventory...")
    inventory = []

    files_to_hash = sorted(
        [
            p for p in out.rglob("*")
            if p.is_file()
            and p.name not in {"FREEZE_INVENTORY.csv", "SHA256SUMS.txt", "FREEZE_COMPLETE.txt"}
        ],
        key=lambda p: p.relative_to(out).as_posix(),
    )

    for i, p in enumerate(files_to_hash, 1):
        rel = p.relative_to(out).as_posix()
        digest = sha256_file(p)
        inventory.append({
            "path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": digest,
        })
        if i % 25 == 0 or i == len(files_to_hash):
            print(f"  hashed {i}/{len(files_to_hash)}")

    write_csv(out / "FREEZE_INVENTORY.csv", inventory)

    with (out / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for r in inventory:
            f.write(f'{r["sha256"]}  {r["path"]}\n')

    complete = f"""EMIT METHANEFUSE FINAL FREEZE COMPLETE

Primary dataset:
  19 pairs
  38 samples
  114 TIFFs

Strict30 sensitivity:
  11 pairs
  22 samples

Canonical metric sets:
  4

Expanded60 AUROC:
  real temporal = {float(eo["auroc"]):.6f}
  t0 repeated   = {float(eb["auroc"]):.6f}
  delta         = {ed["auroc"]:+.6f}

Strict30 AUROC:
  real temporal = {float(so["auroc"]):.6f}
  t0 repeated   = {float(sb["auroc"]):.6f}
  delta         = {sd["auroc"]:+.6f}

STATUS: FREEZE / COMPLETE
"""
    (out / "FREEZE_COMPLETE.txt").write_text(complete, encoding="utf-8")

    # Final sanity
    total_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())

    print("\n" + "=" * 80)
    print("FINAL FREEZE COMPLETE")
    print("=" * 80)
    print("Freeze:", out)
    print("Primary pairs        :", EXPECTED_EXPANDED_PAIRS)
    print("Primary samples      :", EXPECTED_EXPANDED_ROWS)
    print("Physical TIFFs       :", len(actual_tifs))
    print("Strict30 pairs       :", EXPECTED_STRICT_PAIRS)
    print("Canonical metrics    :", len(METRIC_SOURCES))
    print("Freeze size          :", f"{total_bytes / (1024**3):.3f} GiB")
    print("README               :", out / "README_FINAL.md")
    print("FINAL_METRICS        :", out / "FINAL_METRICS.json")
    print("SHA256               :", out / "SHA256SUMS.txt")
    print("\nSTATUS: FREEZE / COMPLETE")


if __name__ == "__main__":
    main()
