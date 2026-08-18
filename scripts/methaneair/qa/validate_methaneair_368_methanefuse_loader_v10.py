#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

HOME = Path.home()
REPO_ROOT = HOME / "MethaneFuse"
DEFAULT_CSV = (
    REPO_ROOT
    / "data"
    / "custom"
    / "methaneair_validated_368_strictqa_eval.csv"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XFORMERS_DISABLED", "1")

from src.data.multisensor import (  # noqa: E402
    TriSensorTemporalCsvDataset,
    custom_collate_fn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load every MethaneAIR-validated S2 control through the MethaneFuse dataset and DataLoader."
    )
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def describe(obj: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []

    if torch.is_tensor(obj):
        lines.append(
            f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
        )
        return lines

    if isinstance(obj, dict):
        lines.append(f"{prefix}dict keys={list(obj.keys())}")
        for key, value in obj.items():
            lines.extend(describe(value, prefix=f"{prefix}  {key}: "))
        return lines

    if isinstance(obj, (list, tuple)):
        lines.append(f"{prefix}{type(obj).__name__} len={len(obj)}")
        for i, value in enumerate(obj[:8]):
            lines.extend(describe(value, prefix=f"{prefix}  [{i}]: "))
        if len(obj) > 8:
            lines.append(f"{prefix}  ...")
        return lines

    lines.append(f"{prefix}{type(obj).__name__}: {obj!r}")
    return lines


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser()

    if not REPO_ROOT.exists():
        raise FileNotFoundError(f"MethaneFuse repo not found: {REPO_ROOT}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Strict eval CSV not found:\n{csv_path}\nRun the v10 build first."
        )

    raw = pd.read_csv(csv_path, low_memory=False)

    if "sensor" in raw.columns:
        raise RuntimeError(
            "Manifest contains forbidden narrow-table 'sensor' column."
        )

    required = [
        "id",
        "sample_id",
        "label",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing wide-table columns: {missing}")

    if args.limit > 0:
        # Write a temporary validation subset because the dataset constructor
        # takes a CSV path rather than an in-memory dataframe.
        raw = raw.head(args.limit).copy()
        temp_csv = csv_path.parent / ".methaneair_loader_validation_subset.csv"
        raw.to_csv(temp_csv, index=False)
        dataset_csv = temp_csv
    else:
        dataset_csv = csv_path
        temp_csv = None

    print("=" * 96)
    print("METHANEAIR 368 METHANEFUSE LOADER VALIDATION")
    print("=" * 96)
    print("CSV:", dataset_csv)
    print("Rows:", len(raw))
    print("Label counts:")
    print(raw["label"].value_counts(dropna=False).sort_index())

    ds = TriSensorTemporalCsvDataset(
        csv_path=str(dataset_csv),
        local_file_cache=None,
        s5p_data_key="ch4",
        s5p_chn_ids_key="chn_ids",
        s5p_channels_last=False,
        align_l89_to_s2=False,
        wv3_chn_ids=None,
        pad_to_multiple=14,
    )

    if hasattr(ds, "max_retries"):
        ds.max_retries = 1

    if len(ds) != len(raw):
        raise RuntimeError(
            f"Dataset length mismatch: len(ds)={len(ds)} CSV={len(raw)}"
        )

    # Fail loudly on the exact row that cannot be read.
    for i in range(len(ds)):
        try:
            _ = ds[i]
        except Exception as exc:
            row_id = raw.iloc[i].get("id", i)
            raise RuntimeError(
                f"Individual loader failure at row {i}, id={row_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if (i + 1) % 50 == 0 or i + 1 == len(ds):
            print(f"Individual rows loaded: {i + 1}/{len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )

    batches = 0
    first_batch = None

    for batch in loader:
        batches += 1
        if first_batch is None:
            first_batch = batch

    expected_batches = (len(ds) + args.batch_size - 1) // args.batch_size

    if batches != expected_batches:
        raise RuntimeError(
            f"DataLoader batch count mismatch: got {batches}, expected {expected_batches}"
        )

    print("\nFIRST BATCH STRUCTURE")
    for line in describe(first_batch):
        print(line)

    print("\n" + "=" * 96)
    print("LOADER VALIDATION PASS")
    print("=" * 96)
    print(f"Rows loaded individually: {len(ds)}")
    print(f"Rows traversed through DataLoader: {len(ds)}")
    print(f"Batches: {batches}")
    print("pad_to_multiple: 14")
    print("custom_collate_fn: PASS")

    if temp_csv is not None:
        try:
            temp_csv.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
