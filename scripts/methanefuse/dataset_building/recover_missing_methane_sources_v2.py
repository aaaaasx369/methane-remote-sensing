#!/usr/bin/env python3
"""
Strictly recover missing methane source tables and Landsat RG metadata.

This version avoids false positives from generated master/output files.

Outputs
-------
outputs/022_strict_recovery_candidates.csv
outputs/023_landsat_rg_source_hits.csv
outputs/024_archive_candidates.csv
outputs/025_strict_recovery_status.csv
outputs/026_landsat_rg_metadata_recovered.csv
outputs/027_unified_methane_master_landsat_recovered.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TEXT_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl", ".md", ".py",
}
ARCHIVE_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z",
}

GENERATED_PATTERNS = [
    "/outputs/000_",
    "/outputs/001_",
    "/outputs/002_",
    "/outputs/003_",
    "/outputs/004_",
    "/outputs/005_",
    "/outputs/006_",
    "/outputs/012_",
    "/outputs/013_",
    "/outputs/014_",
    "/outputs/015_",
    "/outputs/016_",
    "/outputs/017_",
    "/outputs/018_",
    "/outputs/019_",
    "/outputs/020_",
    "/outputs/021_",
    "/outputs/022_",
    "/outputs/023_",
    "/outputs/024_",
    "/outputs/025_",
    "/outputs/026_",
    "/outputs/027_",
    "/outputs/model_ready_subsets/",
    "/results/eval/",
]

TARGETS = {
    "methaneair_baseline_s2_110": {
        "row_counts": {110},
        "required_any_groups": [
            {"methaneair"},
            {"sentinel", "s2"},
            {"image", "path", "tif", "scene", "patch"},
        ],
    },
    "controlled_release_s2_76": {
        "row_counts": {76},
        "required_any_groups": [
            {"controlled", "release"},
            {"sentinel", "s2"},
            {"label", "ground_truth", "release_rate"},
        ],
    },
    "methaneair_observations_435": {
        "row_counts": {435},
        "required_any_groups": [
            {"methaneair"},
            {"date", "time", "acquisition", "flight"},
            {"latitude", "longitude", "location", "site"},
        ],
    },
    "carbonmapper_observations_226": {
        "row_counts": {226, 193},
        "required_any_groups": [
            {"carbon_mapper", "carbonmapper", "carbon", "mapper"},
            {"classification", "tp", "fn", "tn", "fp", "detection"},
            {"emission", "release", "ground_truth"},
        ],
    },
    "historical_multisatellite_17": {
        "row_counts": {17, 44, 88},
        "required_any_groups": [
            {"satellite", "sensor"},
            {"acquisition", "overpass", "timestamp", "date"},
            {"team", "provider", "classification", "result"},
        ],
    },
}

DATE_COLUMNS = [
    "acquisition_time_utc", "acquisition_time", "timestamp", "datetime",
    "date_time", "scene_timestamp", "overpass_time", "date",
    "release_date", "acquisition_date", "image_date",
]

SCENE_COLUMNS = [
    "scene_id", "image_id", "product_id", "granule_id",
    "landsat_scene_id", "source_scene_id",
]

PATH_COLUMNS = [
    "t0_path", "l89_0_path", "image_path", "patch_path", "tif_path",
    "file_path", "filepath", "input_path", "source_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    parser.add_argument(
        "--search-roots",
        nargs="*",
        default=[
            "/project/6002520/yunjung1",
            "/project/def-juliana2/yunjung1",
            "/home/yunjung1",
        ],
    )
    parser.add_argument(
        "--max-text-size-mb",
        type=float,
        default=100.0,
    )
    return parser.parse_args()


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def is_generated(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(pattern in text for pattern in GENERATED_PATTERNS)


def unique_existing_roots(values: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    for value in values:
        path = Path(value).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        key = str(resolved)
        if key in seen or not resolved.exists():
            continue

        seen.add(key)
        roots.append(resolved)

    return roots


def csv_row_count(path: Path) -> int:
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def csv_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as handle:
            return next(csv.reader(handle), [])


def signal_text(path: Path, columns: Iterable[str]) -> str:
    pieces = [
        norm(path.name),
        norm(path.parent.name),
        *[norm(column) for column in columns],
    ]
    return "_".join(pieces)


def group_matches(text: str, group: set[str]) -> bool:
    return any(norm(token) in text for token in group)


def strict_candidate_scan(roots: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    visited: set[str] = set()

    for root in roots:
        for path in root.rglob("*.csv"):
            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)

            if resolved in visited:
                continue
            visited.add(resolved)

            if is_generated(path):
                continue

            try:
                rows = csv_row_count(path)
                columns = csv_header(path)
                text = signal_text(path, columns)

                for target_name, spec in TARGETS.items():
                    if rows not in spec["row_counts"]:
                        continue

                    matches = [
                        group_matches(text, group)
                        for group in spec["required_any_groups"]
                    ]

                    records.append({
                        "target_name": target_name,
                        "path": str(path),
                        "rows": rows,
                        "columns": len(columns),
                        "all_required_groups_match": all(matches),
                        "group_matches": "|".join(
                            "1" if match else "0"
                            for match in matches
                        ),
                        "column_names": "|".join(map(str, columns)),
                        "size_mb": round(
                            path.stat().st_size / 1024**2,
                            4,
                        ),
                    })

            except Exception as exc:
                records.append({
                    "target_name": "scan_error",
                    "path": str(path),
                    "rows": pd.NA,
                    "columns": pd.NA,
                    "all_required_groups_match": False,
                    "group_matches": "",
                    "column_names": "",
                    "size_mb": pd.NA,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    if not records:
        return pd.DataFrame(
            columns=[
                "target_name", "path", "rows", "columns",
                "all_required_groups_match", "group_matches",
                "column_names", "size_mb",
            ]
        )

    result = pd.DataFrame(records)

    sort_columns = [
        column for column in [
            "target_name",
            "all_required_groups_match",
            "rows",
            "path",
        ]
        if column in result.columns
    ]
    ascending = [
        True if column != "all_required_groups_match" else False
        for column in sort_columns
    ]

    return result.sort_values(
        sort_columns,
        ascending=ascending,
    )


def archive_scan(roots: list[Path]) -> pd.DataFrame:
    records = []
    visited: set[str] = set()
    keywords = [
        "methaneair", "controlled", "release", "carbon",
        "mapper", "multisatellite", "satellite", "dataset",
        "backup", "archive", "output",
    ]

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffixes = {suffix.lower() for suffix in path.suffixes}
            if not suffixes.intersection(ARCHIVE_EXTENSIONS):
                continue

            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)

            if resolved in visited:
                continue
            visited.add(resolved)

            text = norm(str(path))
            hits = [keyword for keyword in keywords if keyword in text]

            records.append({
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1024**2, 4),
                "keyword_hits": "|".join(hits),
                "likely_relevant": bool(hits),
            })

    if not records:
        return pd.DataFrame(
            columns=[
                "path", "size_mb", "keyword_hits", "likely_relevant",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["likely_relevant", "size_mb"],
        ascending=[False, False],
    )


def extract_rg_ids(master: pd.DataFrame) -> list[str]:
    if "source_row_ids" not in master.columns:
        return []

    ids = (
        master["source_row_ids"]
        .dropna()
        .astype(str)
        .str.extract(r"(landsat89_\d{3}_RG_[A-Za-z0-9]+)", expand=False)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(ids)


def find_rg_hits(
    roots: list[Path],
    rg_ids: list[str],
    max_text_size_mb: float,
) -> pd.DataFrame:
    if not rg_ids:
        return pd.DataFrame(
            columns=[
                "rg_id", "path", "line_number", "line_preview",
            ]
        )

    records = []
    visited: set[str] = set()

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if is_generated(path):
                continue

            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)

            if resolved in visited:
                continue
            visited.add(resolved)

            try:
                if path.stat().st_size > max_text_size_mb * 1024**2:
                    continue

                with path.open(
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as handle:
                    for line_number, line in enumerate(handle, start=1):
                        matching_ids = [
                            rg_id for rg_id in rg_ids
                            if rg_id in line
                        ]
                        if not matching_ids:
                            continue

                        preview = line.strip().replace("\t", " ")
                        if len(preview) > 500:
                            preview = preview[:500] + "..."

                        for rg_id in matching_ids:
                            records.append({
                                "rg_id": rg_id,
                                "path": str(path),
                                "line_number": line_number,
                                "line_preview": preview,
                            })

            except Exception:
                continue

    if not records:
        return pd.DataFrame(
            columns=[
                "rg_id", "path", "line_number", "line_preview",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["rg_id", "path", "line_number"]
    )


def first_existing_column(
    columns: Iterable[str],
    aliases: Iterable[str],
) -> str | None:
    mapping = {norm(column): column for column in columns}
    for alias in aliases:
        if norm(alias) in mapping:
            return mapping[norm(alias)]
    return None


def parse_date_from_value(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None

    text = str(value)

    patterns = [
        r"LC0[89]_L2S[A-Z]_\d{6}_(20\d{6})_",
        r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)",
        r"(?<!\d)(20\d{6})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue

        raw = "".join(match.groups())
        parsed = pd.to_datetime(
            raw,
            format="%Y%m%d",
            errors="coerce",
            utc=True,
        )
        if pd.notna(parsed):
            return parsed

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )
    if pd.notna(parsed):
        return parsed

    return None


def recover_rg_metadata(
    hits: pd.DataFrame,
    rg_ids: list[str],
) -> pd.DataFrame:
    records = []
    grouped_paths = (
        hits.groupby("path")["rg_id"].apply(set).to_dict()
        if not hits.empty else {}
    )

    for path_text, ids_in_path in grouped_paths.items():
        path = Path(path_text)
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue

        try:
            separator = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=separator)
        except Exception:
            continue

        id_columns = [
            column for column in df.columns
            if any(
                token in norm(column)
                for token in [
                    "sample_id", "source_row", "record_id",
                    "id", "name",
                ]
            )
        ]

        if not id_columns:
            id_columns = list(df.columns)

        date_columns = [
            column for column in df.columns
            if norm(column) in {norm(value) for value in DATE_COLUMNS}
            or any(
                token in norm(column)
                for token in ["date", "time", "timestamp", "acquisition"]
            )
        ]

        scene_columns = [
            column for column in df.columns
            if norm(column) in {norm(value) for value in SCENE_COLUMNS}
            or "scene" in norm(column)
            or "product" in norm(column)
        ]

        path_columns = [
            column for column in df.columns
            if norm(column) in {norm(value) for value in PATH_COLUMNS}
            or any(
                token in norm(column)
                for token in ["path", "tif", "file"]
            )
        ]

        for rg_id in ids_in_path:
            mask = pd.Series(False, index=df.index)

            for column in id_columns:
                values = df[column].astype(str)
                mask = mask | values.str.contains(
                    re.escape(rg_id),
                    regex=True,
                    na=False,
                )

            matched = df[mask]
            if matched.empty:
                continue

            for row_index, row in matched.iterrows():
                parsed_date = None
                date_source_column = None
                date_source_value = None

                for column in [
                    *date_columns,
                    *scene_columns,
                    *path_columns,
                    *id_columns,
                ]:
                    if column not in row.index:
                        continue
                    parsed = parse_date_from_value(row[column])
                    if parsed is not None:
                        parsed_date = parsed
                        date_source_column = column
                        date_source_value = row[column]
                        break

                scene_value = pd.NA
                scene_source_column = pd.NA
                for column in scene_columns:
                    value = row[column]
                    if pd.notna(value) and str(value).strip():
                        scene_value = value
                        scene_source_column = column
                        break

                records.append({
                    "rg_id": rg_id,
                    "source_file": str(path),
                    "source_row_index": row_index,
                    "acquisition_time_utc": (
                        parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                        if parsed_date is not None else pd.NA
                    ),
                    "date_source_column": date_source_column,
                    "date_source_value": date_source_value,
                    "scene_id": scene_value,
                    "scene_source_column": scene_source_column,
                    "matched_columns": "|".join(map(str, df.columns)),
                })

    if not records:
        return pd.DataFrame(
            columns=[
                "rg_id", "source_file", "source_row_index",
                "acquisition_time_utc", "date_source_column",
                "date_source_value", "scene_id",
                "scene_source_column", "matched_columns",
            ]
        )

    output = pd.DataFrame(records)

    # Prefer rows with recovered dates and scenes.
    output["_date_rank"] = output["acquisition_time_utc"].notna().astype(int)
    output["_scene_rank"] = output["scene_id"].notna().astype(int)

    output = (
        output.sort_values(
            ["rg_id", "_date_rank", "_scene_rank"],
            ascending=[True, False, False],
        )
        .drop_duplicates("rg_id", keep="first")
        .drop(columns=["_date_rank", "_scene_rank"])
    )

    return output


def apply_rg_recovery(
    master: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    output = master.copy()

    if recovery.empty or "source_row_ids" not in output.columns:
        return output

    lookup = recovery.set_index("rg_id").to_dict("index")

    for index, value in output["source_row_ids"].items():
        if pd.isna(value):
            continue

        match = re.search(
            r"(landsat89_\d{3}_RG_[A-Za-z0-9]+)",
            str(value),
        )
        if not match:
            continue

        rg_id = match.group(1)
        metadata = lookup.get(rg_id)
        if not metadata:
            continue

        recovered_time = metadata.get("acquisition_time_utc")
        if (
            pd.notna(recovered_time)
            and (
                "acquisition_time_utc" not in output.columns
                or pd.isna(output.at[index, "acquisition_time_utc"])
            )
        ):
            output.at[index, "acquisition_time_utc"] = recovered_time

        recovered_scene = metadata.get("scene_id")
        if (
            pd.notna(recovered_scene)
            and (
                "scene_id" not in output.columns
                or pd.isna(output.at[index, "scene_id"])
            )
        ):
            output.at[index, "scene_id"] = recovered_scene

        site = (
            str(output.at[index, "site_id"])
            if "site_id" in output.columns
            else "unknown_site"
        )
        parsed = pd.to_datetime(
            output.at[index, "acquisition_time_utc"],
            errors="coerce",
            utc=True,
        )

        if pd.notna(parsed):
            output.at[index, "event_group_id"] = (
                f"{site}|{parsed.strftime('%Y-%m-%d')}"
            )

    return output


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    roots = unique_existing_roots(args.search_roots)
    if not roots:
        raise FileNotFoundError("No search roots exist.")

    master_path = outputs / "012_unified_methane_master_fixed.csv"
    if not master_path.exists():
        master_path = outputs / "002_unified_methane_master_dedup.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Master not found: {master_path}")

    master = pd.read_csv(master_path)
    rg_ids = extract_rg_ids(master)

    candidates = strict_candidate_scan(roots)
    candidates_path = outputs / "022_strict_recovery_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    hits = find_rg_hits(
        roots=roots,
        rg_ids=rg_ids,
        max_text_size_mb=args.max_text_size_mb,
    )
    hits_path = outputs / "023_landsat_rg_source_hits.csv"
    hits.to_csv(hits_path, index=False)

    archives = archive_scan(roots)
    archives_path = outputs / "024_archive_candidates.csv"
    archives.to_csv(archives_path, index=False)

    status_records = []
    for target_name in TARGETS:
        subset = candidates[
            (candidates["target_name"] == target_name)
            & candidates["all_required_groups_match"].eq(True)
        ]
        status_records.append({
            "target_name": target_name,
            "strict_match_found": not subset.empty,
            "strict_match_count": len(subset),
            "best_path": (
                str(subset.iloc[0]["path"])
                if not subset.empty else ""
            ),
            "best_rows": (
                subset.iloc[0]["rows"]
                if not subset.empty else pd.NA
            ),
        })

    status = pd.DataFrame(status_records)
    status_path = outputs / "025_strict_recovery_status.csv"
    status.to_csv(status_path, index=False)

    recovery = recover_rg_metadata(hits, rg_ids)
    recovery_path = outputs / "026_landsat_rg_metadata_recovered.csv"
    recovery.to_csv(recovery_path, index=False)

    recovered_master = apply_rg_recovery(master, recovery)
    recovered_master_path = (
        outputs / "027_unified_methane_master_landsat_recovered.csv"
    )
    recovered_master.to_csv(recovered_master_path, index=False)

    landsat = recovered_master[
        recovered_master.get(
            "model_family",
            pd.Series(index=recovered_master.index, dtype="object"),
        )
        .astype("string")
        .eq("landsat_temporal")
    ]

    print("=" * 88)
    print("Strict recovery scan complete")
    print("Search roots:")
    for root in roots:
        print(" -", root)
    print()
    print("Strict source status:")
    print(status.to_string(index=False))
    print()
    print("Landsat RG IDs:", len(rg_ids))
    print("RG source hits:", len(hits))
    print("RG records with recovered date:", int(
        recovery["acquisition_time_utc"].notna().sum()
        if not recovery.empty else 0
    ))
    print("Landsat rows:", len(landsat))
    print("Landsat rows with acquisition time:", int(
        landsat["acquisition_time_utc"].notna().sum()
        if "acquisition_time_utc" in landsat.columns else 0
    ))
    print("Landsat unique events:", int(
        landsat["event_group_id"].nunique(dropna=True)
        if "event_group_id" in landsat.columns else 0
    ))
    print()
    print("Outputs:")
    for path in [
        candidates_path,
        hits_path,
        archives_path,
        status_path,
        recovery_path,
        recovered_master_path,
    ]:
        print(" -", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
