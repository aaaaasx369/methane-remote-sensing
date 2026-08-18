#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

PATH_COLUMNS = ["s2_0_path", "s2_90_path", "s2_360_path"]
REQUIRED_COLUMNS = ["id", "label", "site_id", *PATH_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare five-site leave-one-site-out train/validation/test manifests "
            "for MethaneFuse from the existing 75-row S2 manifest."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/happydoraaa/methane_release_project/outputs/52_methanefuse_smoke_test.csv",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/happydoraaa/MethaneFuse/data/custom/five_site_loso",
    )
    parser.add_argument(
        "--bundle-images",
        action="store_true",
        help=(
            "Copy each unique image into output-root/images and rewrite paths as "
            "relative paths. Recommended before transfer to an Alliance cluster."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-per-class-per-site",
        type=int,
        default=1,
        help=(
            "Rows selected for validation from each label within each training site. "
            "The default keeps the held-out fifth site untouched."
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Input manifest is missing columns: {missing}\n"
            f"Existing columns: {df.columns.tolist()}"
        )


def resolve_existing_path(value: object, input_csv: Path) -> Path:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise FileNotFoundError(f"Empty image path in {input_csv}")

    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                input_csv.parent / path,
                input_csv.parent.parent / path,
                Path("/Users/happydoraaa/methane_release_project") / path,
                Path("/Users/happydoraaa/MethaneFuse") / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve image path: {value}\n"
        + "\n".join(f"  tried: {candidate}" for candidate in candidates)
    )


def provenance_for_site(site_id: str) -> str:
    site_lower = site_id.lower()
    if "casa" in site_lower or "ehrenberg" in site_lower:
        return "controlled_release_derived_label"
    if "methaneair" in site_lower or "ma_site" in site_lower:
        return "methaneair_plume_or_no_known_plume_reference"
    return "unknown_provenance"


def safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join(filter(None, "".join(chars).split("_")))


def stage_image(source: Path, images_dir: Path) -> str:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    destination = images_dir / f"{digest}_{source.name}"
    if not destination.exists():
        shutil.copy2(source, destination)
    return os.path.relpath(destination, start=images_dir.parent)


def select_validation_rows(
    train_pool: pd.DataFrame,
    per_class_per_site: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen_indices: list[int] = []

    for site_id in sorted(train_pool["site_id"].astype(str).unique()):
        site_part = train_pool[train_pool["site_id"].astype(str) == site_id]
        for label in sorted(site_part["label"].astype(int).unique()):
            class_part = site_part[site_part["label"].astype(int) == int(label)]
            max_take = max(0, len(class_part) - 1)
            take = min(per_class_per_site, max_take)
            if take > 0:
                sampled = class_part.sample(
                    n=take,
                    random_state=seed + len(chosen_indices) + int(label),
                )
                chosen_indices.extend(sampled.index.tolist())

    validation = train_pool.loc[sorted(set(chosen_indices))].copy()
    train = train_pool.drop(index=chosen_indices).copy()
    return train, validation


def label_site_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["site_id", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "negative", 1: "positive"})
        .reset_index()
    )
    if "negative" not in table.columns:
        table["negative"] = 0
    if "positive" not in table.columns:
        table["positive"] = 0
    table["total"] = table["negative"] + table["positive"]
    return table[["site_id", "negative", "positive", "total"]]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    require_columns(df)
    df = df.copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["site_id"] = df["site_id"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    bad_labels = sorted(set(df["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"Labels must be binary 0/1, found: {bad_labels}")

    duplicate_ids = df[df["id"].duplicated(keep=False)]
    if len(duplicate_ids):
        raise ValueError(
            "Duplicate id values found:\n"
            + duplicate_ids[["id", "site_id", "label"]].to_string(index=False)
        )

    sites = sorted(df["site_id"].unique().tolist())
    if len(sites) != 5:
        raise ValueError(f"Expected exactly 5 sites, found {len(sites)}: {sites}")

    for column in PATH_COLUMNS:
        df[column] = df[column].apply(
            lambda value: str(resolve_existing_path(value, input_csv))
        )

    df["label_provenance"] = df["site_id"].map(provenance_for_site)
    df["five_site_experiment_scope"] = (
        "exploratory_heterogeneous_label_loso_finetuning"
    )

    if args.bundle_images:
        images_dir = output_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        staged_cache: dict[str, str] = {}
        for column in PATH_COLUMNS:
            rewritten = []
            for value in df[column]:
                if value not in staged_cache:
                    staged_cache[value] = stage_image(Path(value), images_dir)
                rewritten.append(staged_cache[value])
            df[column] = rewritten

    full_manifest = output_root / "all_five_sites.csv"
    df.to_csv(full_manifest, index=False)

    fold_records = []
    audit_records = []

    for fold_index, held_out_site in enumerate(sites):
        fold_slug = f"fold_{fold_index}_{safe_slug(held_out_site)}"
        fold_dir = output_root / fold_slug
        fold_dir.mkdir(parents=True, exist_ok=True)

        test = df[df["site_id"] == held_out_site].copy()
        train_pool = df[df["site_id"] != held_out_site].copy()
        train, validation = select_validation_rows(
            train_pool=train_pool,
            per_class_per_site=args.validation_per_class_per_site,
            seed=args.seed + fold_index * 100,
        )

        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Empty split in {fold_slug}: "
                f"train={len(train)}, val={len(validation)}, test={len(test)}"
            )

        assert held_out_site not in set(train["site_id"])
        assert held_out_site not in set(validation["site_id"])
        assert set(test["site_id"]) == {held_out_site}

        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        test_path = fold_dir / "test.csv"
        train.to_csv(train_path, index=False)
        validation.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        fold_records.append(
            {
                "fold_index": fold_index,
                "fold_slug": fold_slug,
                "held_out_site": held_out_site,
                "train_csv": os.path.relpath(train_path, start=output_root),
                "val_csv": os.path.relpath(val_path, start=output_root),
                "test_csv": os.path.relpath(test_path, start=output_root),
                "train_rows": len(train),
                "val_rows": len(validation),
                "test_rows": len(test),
            }
        )

        for split_name, split_df in [
            ("train", train),
            ("val", validation),
            ("test", test),
        ]:
            for _, row in label_site_table(split_df).iterrows():
                audit_records.append(
                    {
                        "fold_index": fold_index,
                        "fold_slug": fold_slug,
                        "held_out_site": held_out_site,
                        "split": split_name,
                        **row.to_dict(),
                    }
                )

    folds = pd.DataFrame(fold_records)
    folds.to_csv(output_root / "folds.csv", index=False)
    folds[
        ["fold_index", "fold_slug", "held_out_site", "train_csv", "val_csv", "test_csv"]
    ].to_csv(output_root / "folds.tsv", sep="\t", index=False, header=False)

    pd.DataFrame(audit_records).to_csv(output_root / "split_audit.csv", index=False)

    metadata = {
        "input_manifest": str(input_csv),
        "rows": int(len(df)),
        "sites": sites,
        "label_counts": {
            str(key): int(value)
            for key, value in df["label"].value_counts().sort_index().items()
        },
        "bundle_images": bool(args.bundle_images),
        "validation_per_class_per_site": int(args.validation_per_class_per_site),
        "seed": int(args.seed),
        "important_limitations": [
            "The five sites do not share one uniform physical ground-truth definition.",
            "MethaneAIR negatives are no-known-plume references, not confirmed zero-emission records.",
            "The current S2 temporal slots may repeat one acquisition and use synthetic 12-band conversion.",
            "This is an exploratory five-site heterogeneous-label adaptation experiment.",
        ],
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nFull five-site audit:")
    print(label_site_table(df).to_string(index=False))
    print("\nFold summary:")
    print(folds.to_string(index=False))
    print("\nCreated:")
    for path in [
        full_manifest,
        output_root / "folds.csv",
        output_root / "folds.tsv",
        output_root / "split_audit.csv",
        output_root / "experiment_metadata.json",
    ]:
        print(path)


if __name__ == "__main__":
    main()
